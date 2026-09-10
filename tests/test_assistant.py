"""The Jarvis layer: reference resolution, the local intent grammar, the speech
queue, and the end-to-end assistant driving a simulated case."""
import time

import numpy as np
import pytest

from bridge.assistant.announce import Announcer
from bridge.assistant.conversation import Conversation, extract_subject
from bridge.assistant.intents import parse
from bridge.voice.assistant import is_echo


# --- conversation memory ----------------------------------------------------------------
def test_pronouns_resolve_against_the_last_subject():
    c = Conversation()
    c.remember("where is the metzenbaum scissors", "Here.", subject="metzenbaum scissors")
    assert c.resolve("point to it")[0] == "point to the metzenbaum scissors"
    assert c.resolve("where is it")[0] == "where is the metzenbaum scissors"
    assert c.resolve("show me that one")[0] == "show the metzenbaum scissors"


def test_plural_only_instrument_names_are_not_mangled():
    """Half a tray is plural-only. 'Pass me the forcep' is not a sentence."""
    from bridge.perception.scene_graph import normalise_label

    assert normalise_label("metzenbaum scissors") == "metzenbaum scissors"
    assert normalise_label("toothed forceps") == "toothed forceps"
    assert normalise_label("artery clamps") == "artery clamp"      # genuinely plural


def test_the_other_one_asks_for_a_different_instance():
    c = Conversation()
    c.remember("where is the artery clamp", "Here.", subject="artery clamp", entity_ids=["e1"])
    text, another = c.resolve("the other one")
    assert another and "other artery clamp" in text
    text, another = c.resolve("show me another one")
    assert another and "artery clamp" in text
    assert c.mentioned == ["e1"]


def test_a_fresh_noun_does_not_get_rewritten():
    c = Conversation()
    c.remember("where is the scalpel", "Here.", subject="scalpel")
    assert c.resolve("where is the gauze")[0] == "where is the gauze"
    assert c.resolve("start the closing count")[0] == "start the closing count"


def test_a_stale_subject_is_not_used():
    c = Conversation(subject_ttl_s=0.05)
    c.remember("where is the scalpel", subject="scalpel")
    time.sleep(0.08)
    assert c.active_subject == ""
    assert c.resolve("point to it")[0] == "point to it"


def test_subject_extraction_strips_the_command_verb():
    assert extract_subject("where is the curved artery clamp?") == "curved artery clamp"
    assert extract_subject("project the number 15 blade") == "number 15 blade"
    assert extract_subject("hand me the needle holder please") == "needle holder"
    assert extract_subject("show me the other lap pad") == "lap pad"


def test_context_block_carries_the_case_and_recent_turns():
    c = Conversation()
    c.case_context = "laparotomy, closing count"
    c.remember("where is the scalpel", "Here.", subject="scalpel")
    block = c.context_block()
    assert "laparotomy, closing count" in block
    assert "user: where is the scalpel" in block
    assert "scalpel" in block.split("refers to:")[-1]


# --- intent grammar ----------------------------------------------------------------------
@pytest.mark.parametrize("text,target", [
    ("project the scalpel", "scalpel"),
    ("Project the number 15 blade", "number 15 blade"),
    ("where is the needle holder", "needle holder"),
    ("show me the lap pads", "lap pads"),
    ("point to the sharps container", "sharps container"),
    ("highlight the metzenbaum on the tray", "metzenbaum"),
    ("hand me the toothed forceps", "toothed forceps"),
    ("light up the suction tip", "suction tip"),
])
def test_locate_phrasings_are_recognised_locally(text, target):
    i = parse(text)
    assert i.kind == "locate" and i.target == target


def test_locate_refuses_to_project_chatter():
    assert parse("show me").kind != "locate"
    assert parse("where is it").kind != "locate"          # a pronoun must be resolved first


@pytest.mark.parametrize("text,kind", [
    ("start the count", "count_begin"),
    ("count in", "count_begin"),
    ("closing count", "count_phase"),
    ("final count", "count_phase"),
    ("what's the count", "count_status"),
    ("how many lap pads", "count_status"),
    ("what's missing", "whats_missing"),
    ("what do you see", "scene_describe"),
    ("start a case", "case_start"),
    ("close the case", "case_close"),
    ("where are we", "case_status"),
    ("quiet", "mute"),
    ("alerts on", "unmute"),
    ("clear", "clear"),
    ("help", "help"),
])
def test_case_and_count_control_words(text, kind):
    assert parse(text).kind == kind


def test_bare_numbers_are_only_counts_while_counting():
    assert parse("four artery clamps").kind == "unknown"           # not in a count: ambiguous
    i = parse("four artery clamps", case_phase="count_in", counting=True)
    assert i.kind == "count_record" and i.target == "artery clamp" and i.quantity == 4
    i = parse("lap pads ten", case_phase="count_final", counting=True)
    assert i.kind == "count_record" and i.quantity == 10


def test_additions_to_the_field_are_their_own_intent():
    i = parse("adding two lap pads", case_phase="procedure")
    assert i.kind == "count_add" and i.target == "lap pad" and i.quantity == 2
    i = parse("opening another raytec", case_phase="procedure")
    assert i.kind == "count_add" and i.quantity == 1


def test_yes_and_skip_only_count_inside_a_checklist():
    assert parse("yes").kind == "unknown"
    assert parse("yes", in_checklist=True).kind == "case_confirm"
    assert parse("not applicable", in_checklist=True).kind == "case_skip"
    assert parse("skip", in_checklist=True).kind == "case_skip"


def test_open_ended_requests_fall_through_to_the_model():
    assert parse("what colour is the tube with the yellow cap").kind == "unknown"
    assert parse("is the tray laid out the way the surgeon likes").kind == "unknown"


# --- speech queue ------------------------------------------------------------------------
def test_announcer_speaks_in_priority_order():
    """Three utterances queued while the voice is busy come out most-urgent first."""
    import threading

    said, busy = [], threading.Event()

    def speak(text):
        busy.wait(2.0)          # the first utterance holds the channel while the rest queue
        said.append(text)

    a = Announcer(speak)
    a.start()
    try:
        a.say("holding the channel", "info")
        time.sleep(0.1)
        a.say("background note", "info")
        a.say("answer to your question", "reply")
        a.say("caution about a sharp", "caution")
        busy.set()
        deadline = time.time() + 3.0
        while len(said) < 4 and time.time() < deadline:
            time.sleep(0.02)
    finally:
        a.stop()
    assert said == ["holding the channel", "answer to your question",
                    "caution about a sharp", "background note"]


def test_critical_preempts_and_clears_the_queue():
    said, stops = [], []
    a = Announcer(said.append, stop_speaking=lambda: stops.append(1))
    a.start()
    try:
        a.say("a long tray description", "info")
        a.say("Count discrepancy. Do not close.", "critical")
        time.sleep(0.4)
    finally:
        a.stop()
    assert stops, "critical alert must cut off whatever is being said"
    assert "Count discrepancy. Do not close." in said


def test_duplicate_announcements_are_dropped():
    said = []
    a = Announcer(said.append, dedupe_s=30.0)
    assert a.say("Two lap pads are not on the tray.", "caution")
    assert not a.say("Two lap pads are not on the tray.", "caution")
    assert len(said) == 1


def test_stale_cautions_are_dropped_rather_than_spoken_late():
    said = []
    a = Announcer(said.append, stale_after_s=0.05)
    a.say("this is already out of date", "caution")
    a.spoken.clear()
    said.clear()
    a._q.put(type(a._q.queue[0]) if False else None) if False else None  # no-op guard
    # queue an item by hand with an old timestamp
    from bridge.assistant.announce import _Item
    a._q.put(_Item(2, 99, "old caution", "caution", ts=time.time() - 10))
    a._drain_sync()
    assert said == []


def test_echo_of_our_own_voice_is_not_treated_as_a_command():
    spoken = "Count discrepancy. Two raytec sponges short. Recount before closing."
    assert is_echo("count discrepancy two raytec sponges short", spoken)
    assert not is_echo("stop", spoken)
    assert not is_echo("show me the other clamp", spoken)


# --- end to end on the simulator -----------------------------------------------------------
@pytest.fixture
def core(tmp_path):
    from bridge.app.application import BridgeCore
    from bridge.app.config import ConfigManager

    cfg = ConfigManager(tmp_path / "settings.yaml")
    cfg.settings.profiles_dir = tmp_path / "profiles"
    cfg.settings.log_dir = tmp_path / "logs"
    cfg.settings.assistant.records_dir = tmp_path / "records"
    cfg.settings.assistant.ai_label_interval_s = 1e6      # no background AI during tests
    c = BridgeCore(cfg)
    c.enter_simulation(seed=3, scene="workshop")
    c.calibration.method.settle_s = 0.02
    c.cameras.wait_for_frame(3.0)
    yield c
    c.shutdown()


def test_assistant_is_wired_into_the_core(core):
    assert core.assistant is not None
    assert core.handle_spoken("help").startswith("Say: project")


def test_project_resolves_from_scene_memory_without_calling_the_model(core):
    a = core.assistant
    assert core.calibrate().success
    world, proj, cam = core.simulation
    frame = core.cameras.wait_for_frame(2.0)
    a.on_frame(frame)
    gt = cam.object_bbox_in_camera(world.get("obj-screwdriver"))
    a.scene.apply_ai([("screwdriver", gt, 0.95, "red handled")])

    calls = {"n": 0}
    original = core.ai.identify_target

    def counting(*args, **kw):
        calls["n"] += 1
        return original(*args, **kw)

    core.ai.identify_target = counting
    reply = core.handle_spoken("project the screwdriver")
    assert "screwdriver" in reply.lower()
    assert calls["n"] == 0, "a known object must not cost an AI round-trip"
    assert core.executor.last_states
    st = core.executor.last_states[0]
    assert st.center.distance_to(gt.center) < 40


def test_unknown_object_falls_back_to_the_model_and_is_then_remembered(core):
    a = core.assistant
    assert core.calibrate().success
    core.cameras.wait_for_frame(2.0)
    assert a.scene.find_one("screwdriver") is None
    reply = core.handle_spoken("where is the screwdriver")
    assert reply and "could not find" not in reply.lower()
    assert a.scene.find_one("screwdriver") is not None      # learned from the model's answer


def test_a_whole_case_runs_hands_free(core, tmp_path):
    a = core.assistant
    assert core.calibrate().success
    core.cameras.wait_for_frame(2.0)

    assert "briefing" in core.handle_spoken("start a case for a minor set").lower() or a.phase == "briefing"
    assert a.counts.set is not None and a.counts.set.id == "minor"

    a.case.go_to("timeout")
    n = len(a.case.current_phase_items)
    for _ in range(n):
        core.handle_spoken("confirmed")
    assert a.phase != "timeout"

    core.handle_spoken("start the count")
    assert a.phase == "count_in"
    reply = core.handle_spoken("count as per the sheet")
    assert "reconciles" in reply and "own count sheet" in reply

    a.case.go_to("procedure")
    reply = core.handle_spoken("adding two lap pads")
    assert "now expects" in reply
    assert a.counts.line("laparotomy pad").baseline == 7

    core.handle_spoken("final count")
    assert a.phase == "count_final"
    reply = core.handle_spoken("six laparotomy pads")
    assert "missing" in reply.lower()
    assert a.counts.line("laparotomy pad").status_at("final") == "short"

    reply = core.handle_spoken("seven laparotomy pads")
    assert "Correct" in reply
    for line in a.counts.lines:
        if line.counted_final is None:
            line.counted_final = line.baseline
    reply = core.handle_spoken("close the case")
    assert "reconciles" in reply and "Record saved" in reply
    assert (tmp_path / "records").exists()


def test_follow_up_questions_use_conversation_memory(core):
    a = core.assistant
    assert core.calibrate().success
    frame = core.cameras.wait_for_frame(2.0)
    world, proj, cam = core.simulation
    screws = [o for o in world.objects if o.label == "screw"]
    a.on_frame(frame)
    a.scene.apply_ai([("screw", cam.object_bbox_in_camera(o), 0.9, "small metal screw") for o in screws])

    first = core.handle_spoken("where is the screw")
    assert "screw" in first.lower()
    shown_first = list(a.conversation.mentioned)
    assert shown_first

    second = core.handle_spoken("the other one")
    assert second and "screw" in second.lower()
    assert len(a.conversation.mentioned) > len(shown_first), "a different instance must be shown"


def test_dosing_questions_are_still_refused_through_the_assistant(core):
    assert core.calibrate().success
    core.cameras.wait_for_frame(2.0)
    reply = core.handle_spoken("how much adrenaline should I give")
    assert "dosing" in reply.lower() or "clinician" in reply.lower()


def test_whats_missing_marks_the_empty_slot(core):
    """Two clamps counted in, one gone: BRIDGE names it and outlines where it was."""
    from bridge.spatial.geometry import BoundingBox
    from bridge.vision.detection import Detection

    def det(x, y):
        return Detection.from_bbox("", BoundingBox(x=x, y=y, w=40, h=40), 0.9, "contour")

    a = core.assistant
    assert core.calibrate().success
    core.handle_spoken("start a case")
    a.case.go_to("count_in")

    a.scene.absent_after_s = 0.0
    a.scene.observe([det(100, 100), det(300, 100), det(500, 300)], now=0.0)
    a.scene.apply_ai([("artery clamp", BoundingBox(x=100, y=100, w=40, h=40), 0.9, ""),
                      ("artery clamp", BoundingBox(x=300, y=100, w=40, h=40), 0.9, ""),
                      ("gauze", BoundingBox(x=500, y=300, w=40, h=40), 0.9, "")])
    a.counts.record("artery clamp", 2, "initial")
    a.counts.observe(a.scene)
    assert a._capture_layout() >= 2

    a.scene.observe([det(100, 100), det(500, 300)], now=5.0)     # one clamp taken
    a.counts.observe(a.scene)
    assert a.counts.line("artery clamp").observed == 1

    reply = a.whats_missing()
    assert "artery clamp" in reply.lower()
    assert "marked where" in reply.lower()
    from bridge.render.overlay import GROUP_GAPS
    assert any(p.group == GROUP_GAPS for p in core.renderer.scene.items())


def test_muting_stops_proactive_speech_but_not_answers(core):
    a = core.assistant
    reply = core.handle_spoken("quiet")
    assert not a.proactive and "muted" in reply
    assert core.handle_spoken("what do you see")
    assert "back on" in core.handle_spoken("alerts on")
    assert a.proactive


def test_snapshot_gives_the_ui_everything_in_one_read(core):
    a = core.assistant
    core.handle_spoken("start a case for a minor set")
    a.case.go_to("count_in")
    a.counts.accept_expected("initial")
    snap = a.snapshot()
    assert snap["phase"] == "count_in" and snap["set"] == "Minor / basic set"
    assert snap["counted"] == snap["expected"] > 0
    assert snap["verdict"] == "reconciled"
    assert isinstance(snap["board"], list) and snap["board"]
