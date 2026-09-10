"""Case autopilot (WHO-style checklist) and the proactive safety monitor."""
import json

import pytest

from bridge.perception.scene_graph import SceneGraph
from bridge.spatial.geometry import BoundingBox
from bridge.surgical.case import PHASE_ORDER, CaseSession
from bridge.surgical.counts import CountSession
from bridge.surgical.monitor import SafetyMonitor
from bridge.surgical.sets import SetLibrary
from bridge.surgical.trays import TrayLayout, Zone
from bridge.vision.detection import Detection


def det(x, y, w=40, h=40, label=""):
    return Detection.from_bbox(label, BoundingBox(x=x, y=y, w=w, h=h), 0.9, "test")


@pytest.fixture
def case():
    return CaseSession(instrument_set=SetLibrary().get("minor"), case_id="c1", procedure_name="appendicectomy")


# --- case autopilot ------------------------------------------------------------------------
def test_case_walks_the_phases_in_order(case):
    case.start()
    assert case.phase == "briefing"
    seen = [case.phase]
    for _ in range(12):
        case.next_phase()
        seen.append(case.phase)
        if case.phase == "closed":
            break
    assert seen[:4] == ["briefing", "sign_in", "timeout", "count_in"]
    assert seen.index("count_in") < seen.index("procedure") < seen.index("count_final")
    assert PHASE_ORDER.index("sign_out") > PHASE_ORDER.index("count_final")


def test_checklist_items_are_read_one_at_a_time_and_recorded(case):
    case.start()
    case.go_to("timeout")
    first = case.current_item
    assert first is not None and "stop" in case.announce().lower()
    n = len(case.current_phase_items)
    for _ in range(n):
        case.confirm("yes")
    assert case.phase != "timeout"                     # moved on automatically
    keys = [c.key for c in case.confirmations if c.phase == "timeout"]
    assert len(keys) == n and "antibiotic" in keys


def test_skipping_a_critical_item_is_recorded_not_blocked(case):
    case.start()
    case.go_to("sign_in")
    reply = case.skip("no consent form yet")
    assert "Recorded as not done" in reply           # BRIDGE never refuses to proceed
    assert case.critical_skips and case.critical_skips[0].key == "identity"


def test_case_phase_drives_the_count_phase(case):
    case.start()
    case.go_to("count_in")
    assert case.counts.phase == "initial"
    case.go_to("procedure")
    assert case.counts.phase == "added"
    case.go_to("count_final")
    assert case.counts.phase == "final"


def test_closing_a_case_reports_the_arithmetic_and_the_skips(case, tmp_path):
    case.start()
    case.go_to("sign_in")
    case.skip()                                        # a critical item not confirmed
    case.go_to("count_in")
    case.counts.accept_expected("initial")
    case.go_to("count_final")
    for line in case.counts.lines:
        line.counted_final = line.baseline
    case.counts.line("raytec sponge").counted_final = 9   # one short
    rec, spoken = case.close()
    assert rec.verdict == "discrepancy"
    assert "Do not close" in spoken and "critical checklist item" in spoken
    assert case.phase == "closed"

    path = case.save(tmp_path)
    data = json.loads(path.read_text())
    assert data["procedure"] == "appendicectomy"
    assert data["critical_skips"] == ["identity"]
    assert "not a medical device" not in data["disclaimer"].lower()   # wording check
    assert "responsibility of the operating team" in data["disclaimer"]
    assert "final" in [r["phase"] for r in data["counts"]["reconciliations"]]
    assert "CASE c1" in case.report_text()


def test_back_steps_within_a_phase_then_to_the_previous_phase(case):
    case.start()
    case.go_to("timeout")
    case.confirm("yes")
    case.confirm("yes")
    assert case.progress()[0] == 2
    case.back()
    assert case.progress()[0] == 1
    case.back()
    case.back()                                         # already at the top: go back a phase
    assert case.phase == "sign_in"


# --- proactive monitor ----------------------------------------------------------------------
def make_scene(boxes, labels=(), t=0.0):
    g = SceneGraph()
    g.observe([det(*b) for b in boxes], now=t)
    if labels:
        g.apply_ai([(name, BoundingBox(x=b[0], y=b[1], w=40, h=40), 0.9, "")
                    for name, b in zip(labels, boxes)])
    return g


def test_monitor_says_nothing_when_nothing_is_wrong():
    m = SafetyMonitor()
    scene = make_scene([(100, 100), (200, 100)])
    assert m.evaluate(scene, None, "procedure", TrayLayout(), None, now=0.0) == []


def counted_and_visible(set_id="minor"):
    """A session where everything is counted in and every item is visible on the tray."""
    counts = CountSession(SetLibrary().get(set_id))
    counts.accept_expected("initial")
    for line in counts.lines:
        line.observed = line.baseline
    return counts


def test_missing_sponge_at_closing_is_critical_and_projected():
    counts = counted_and_visible()
    counts.line("raytec sponge").observed = 7          # only 7 of 10 visible
    m = SafetyMonitor(absence_grace_s=0.0)
    scene = make_scene([(100, 100)])
    alerts = m.evaluate(scene, counts, "count_final", TrayLayout(), None, now=10.0)
    keys = {a.key: a for a in alerts}
    a = keys["closing_short:raytec sponge"]
    assert a.level == "critical" and a.project and a.speak
    assert "3 raytec sponge not visible" in a.text
    assert "Account for it before closing" in a.text
    # BRIDGE reports what it can see; it does not declare a retained item
    assert "retained" not in a.text.lower()


def test_the_same_alert_is_not_repeated_while_it_persists():
    counts = counted_and_visible()
    counts.line("raytec sponge").observed = 7
    m = SafetyMonitor(absence_grace_s=0.0, cooldown_s=60.0)
    scene = make_scene([(100, 100)])
    first = m.evaluate(scene, counts, "count_final", TrayLayout(), None, now=10.0)
    again = m.evaluate(scene, counts, "count_final", TrayLayout(), None, now=12.0)
    later = m.evaluate(scene, counts, "count_final", TrayLayout(), None, now=80.0)
    assert first and not again and later          # once, quiet, then again after the cooldown


def test_a_sharp_off_the_tray_earns_a_warning_only_after_the_grace_period():
    layout = TrayLayout()
    layout.set_zone(Zone.rect("tray", 0, 0, 400, 400, kind="tray"))
    layout.set_zone(Zone.rect("neutral", 420, 0, 100, 100, kind="neutral"))
    scene = make_scene([(800, 800)], labels=["scalpel blade"])
    m = SafetyMonitor(sharp_grace_s=20.0)
    assert m.evaluate(scene, None, "procedure", layout, None, now=0.0) == []      # grace
    alerts = m.evaluate(scene, None, "procedure", layout, None, now=25.0)
    assert alerts and alerts[0].key.startswith("sharp_loose:")
    assert "outside the tray and the neutral zone" in alerts[0].text
    # a sharp in the neutral zone is fine
    ok = make_scene([(440, 40)], labels=["scalpel blade"])
    assert SafetyMonitor(sharp_grace_s=0.0).evaluate(ok, None, "procedure", layout, None, now=30.0) == []


def test_everything_vanishing_at_once_reads_as_a_blocked_camera():
    g = SceneGraph(absent_after_s=0.2)
    g.observe([det(100 * i, 100) for i in range(1, 7)], now=0.0)
    delta = g.observe([], now=5.0)                # a drape thrown over the tray
    assert len(delta.disappeared) == 6
    m = SafetyMonitor(absence_grace_s=0.0)
    alerts = m.evaluate(g, None, "procedure", TrayLayout(), delta, now=5.0)
    assert any(a.key == "view_blocked" for a in alerts)
    assert "blocking the camera" in alerts[0].text


def test_muting_silences_one_rule_without_disabling_the_monitor():
    counts = counted_and_visible()
    counts.line("raytec sponge").observed = 7
    m = SafetyMonitor(absence_grace_s=0.0)
    m.mute("closing_short:raytec sponge")
    scene = make_scene([(100, 100)])
    assert m.evaluate(scene, counts, "count_final", TrayLayout(), None, now=10.0) == []
    m.unmute()
    assert m.evaluate(scene, counts, "count_final", TrayLayout(), None, now=20.0)


def test_critical_alerts_bypass_the_rate_limit():
    counts = counted_and_visible()
    counts.line("raytec sponge").observed = 7
    counts.line("scalpel blade").observed = 0
    m = SafetyMonitor(absence_grace_s=0.0, min_interval_s=30.0)
    alerts = m.evaluate(counts=counts, scene=make_scene([(100, 100)]), phase="count_final",
                        layout=TrayLayout(), now=10.0)
    assert len(alerts) == 2 and all(a.level == "critical" for a in alerts)
