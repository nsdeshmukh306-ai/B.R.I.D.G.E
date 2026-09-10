"""The count engine — the arithmetic that has to be right."""
import json

import pytest

from bridge.perception.scene_graph import SceneGraph
from bridge.spatial.geometry import BoundingBox
from bridge.surgical.counts import CountSession, parse_spoken_count
from bridge.surgical.sets import SetLibrary, classify
from bridge.surgical.trays import TrayLayout, Zone, infer_tray_zone
from bridge.vision.detection import Detection


def det(label, x, y, w=40, h=40):
    return Detection.from_bbox(label, BoundingBox(x=x, y=y, w=w, h=h), 0.9, "test")


@pytest.fixture
def session():
    return CountSession(SetLibrary().get("minor"), case_id="test-case")


# --- taxonomy ---------------------------------------------------------------------------
def test_countable_items_are_categorised_by_name():
    assert classify("laparotomy pad") == "sponge"
    assert classify("raytec sponge") == "sponge"
    assert classify("scalpel blade") == "sharp"
    assert classify("suture needle") == "sharp"
    assert classify("vessel loop") == "miscellaneous"
    assert classify("babcock forceps") == "instrument"


def test_spoken_numbers_are_parsed_either_way_round():
    assert parse_spoken_count("four artery clamps") == ("artery clamp", 4)
    assert parse_spoken_count("artery clamps four") == ("artery clamp", 4)
    assert parse_spoken_count("lap pads 10") == ("lap pad", 10)
    assert parse_spoken_count("no sponges") == ("sponge", 0)
    assert parse_spoken_count("scalpel handle") is None       # no number: not a count


# --- the core rule ----------------------------------------------------------------------
def test_final_count_must_equal_initial_plus_additions(session):
    session.set_phase("initial")
    session.accept_expected("initial")             # whole tray counted as laid out
    assert session.reconcile("initial").verdict == "reconciled"
    assert session.line("raytec sponge").counted_initial == 10

    session.set_phase("added")
    session.add_item("raytec sponge", 5)          # a fresh packet opened mid-case
    line = session.line("raytec sponge")
    assert line.baseline == 15

    session.set_phase("final")
    for l in session.lines:                        # everything else accounted for
        l.counted_final = l.baseline
    session.record("raytec sponge", 14)            # one unaccounted for
    rec = session.reconcile("final")
    assert rec.verdict == "discrepancy"
    assert [l.name for l in rec.short] == ["raytec sponge"]
    assert "Do not close" in rec.spoken()
    assert session.unaccounted() == [(line, 1)]
    assert session.retained_risk

    session.record("raytec sponge", 15)           # found on the drape
    rec = session.reconcile("final")
    assert rec.verdict == "reconciled" and not session.unaccounted()


def test_a_reconciled_count_never_authorises_closure(session):
    session.accept_expected("initial")
    rec = session.reconcile("initial")
    text = rec.spoken()
    assert "reconciles" in text
    assert "own count sheet" in text              # the decision stays with the team
    assert "safe to close" not in text.lower()


def test_extra_items_are_flagged_as_loudly_as_missing_ones(session):
    session.accept_expected("initial")
    session.set_phase("final")
    for l in session.lines:
        l.counted_final = l.baseline
    session.record("scalpel blade", 3)             # baseline is 2
    rec = session.reconcile("final")
    assert rec.verdict == "discrepancy" and rec.over and "extra" in rec.spoken()


def test_pending_lines_block_a_verdict(session):
    session.set_phase("initial")
    session.record("raytec sponge", 10)
    rec = session.reconcile("initial")
    assert rec.verdict == "pending" and rec.pending
    assert "incomplete" in rec.spoken()


def test_unknown_items_can_be_added_to_the_sheet_mid_case(session):
    line, ack = session.record("bulldog clamp", 2, "initial")
    assert line is not None and line.name == "bulldog clamp"
    assert line.category == "miscellaneous"
    assert "Recorded" in ack


def test_a_similar_name_never_lands_on_the_wrong_count_line(session):
    """'bulldog clamp' shares a noun with 'artery clamp'. Merging them corrupts the count."""
    assert session.line("bulldog clamp") is None
    assert session.line("clamp").name == "artery clamp"           # a shorter way of saying it
    assert session.line("curved artery clamp").name == "artery clamp"
    assert session.line("mosquito").name == "artery clamp"        # alias
    assert session.line("lap pad").name == "laparotomy pad"
    assert session.line("suture scissors") is None                # not the mayo or the metz


def test_accept_expected_records_the_whole_sheet(session):
    rec = session.accept_expected("initial")
    assert rec.verdict == "reconciled"
    assert all(l.counted_initial == l.expected for l in session.lines)
    counted, expected = rec.totals
    assert counted == expected == session.set.total


def test_closing_short_speaks_the_discrepancy_protocol(session):
    session.accept_expected("initial")
    session.set_phase("final")
    for l in session.lines:
        l.counted_final = l.baseline
    session.line("laparotomy pad").counted_final = 3        # two missing
    rec = session.reconcile("final")
    text = rec.spoken()
    assert "2 laparotomy pad short" in text
    assert "search the field" in text


# --- vision is advisory -------------------------------------------------------------------
def test_camera_tally_is_separate_from_the_human_count(session):
    scene = SceneGraph()
    scene.observe([det("", 100, 100), det("", 200, 100), det("", 300, 100)], now=0.0)
    scene.apply_ai([("raytec sponge", BoundingBox(x=100, y=100, w=40, h=40), 0.9, ""),
                    ("raytec sponge", BoundingBox(x=200, y=100, w=40, h=40), 0.9, ""),
                    ("scalpel blade", BoundingBox(x=300, y=100, w=40, h=40), 0.9, "")])
    session.observe(scene)
    assert session.line("raytec sponge").observed == 2
    assert session.line("scalpel blade").observed == 1
    # observed never becomes the count
    assert session.line("raytec sponge").counted_initial is None
    disagreements = dict((l.name, (obs, exp)) for l, obs, exp in session.visual_disagreements())
    assert disagreements["raytec sponge"] == (2, 10)


def test_record_is_serialisable_and_carries_the_whole_case(session, tmp_path):
    session.accept_expected("initial")
    session.add_item("raytec sponge", 2)
    session.set_phase("final")
    session.record("raytec sponge", 12)
    session.close()
    path = session.save(tmp_path)
    data = json.loads(path.read_text())
    assert data["case_id"] == "test-case" and data["closed"] is True
    assert data["set"] == "minor"
    phases = [r["phase"] for r in data["reconciliations"]]
    assert "final" in phases
    assert any(e["kind"] == "added" for e in data["events"])


def test_a_closed_case_refuses_further_counts(session):
    session.accept_expected("initial")
    session.close()
    line, ack = session.record("raytec sponge", 9)
    assert line is None and "already closed" in ack


# --- tray geography --------------------------------------------------------------------------
def test_tray_layout_marks_the_slot_a_missing_item_came_from():
    scene = SceneGraph(absent_after_s=0.4)
    scene.observe([det("", 100, 100), det("", 300, 100)], now=0.0)
    scene.apply_ai([("artery clamp", BoundingBox(x=100, y=100, w=40, h=40), 0.9, ""),
                    ("artery clamp", BoundingBox(x=300, y=100, w=40, h=40), 0.9, "")])
    layout = TrayLayout()
    assert layout.capture(scene) == 2
    assert layout.item_names() == ["artery clamp"]

    scene.observe([det("", 100, 100)], now=1.0)            # one clamp taken
    empty = layout.empty_slots(scene, "artery clamp")
    assert len(empty) == 1
    assert empty[0].position.x == pytest.approx(320, abs=5)
    assert len(empty[0].outline()) == 4
    assert len(layout.occupied_slots(scene)) == 1


def test_tray_zone_is_inferred_and_answers_on_or_off_the_tray():
    scene = SceneGraph()
    scene.observe([det("", 100, 100), det("", 300, 120), det("", 200, 260)], now=0.0)
    tray = infer_tray_zone(scene)
    assert tray is not None and tray.kind == "tray"
    assert tray.contains(BoundingBox(x=200, y=180, w=20, h=20))
    assert not tray.contains(BoundingBox(x=900, y=900, w=20, h=20))
    layout = TrayLayout()
    layout.set_zone(tray)
    layout.set_zone(Zone.rect("neutral", 600, 100, 200, 200, kind="neutral"))
    assert layout.zone("neutral").contains(BoundingBox(x=650, y=150, w=10, h=10))
