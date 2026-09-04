"""Voice pipeline (synthetic audio), healthcare safety, procedures and guided flow."""
import time

import numpy as np
import pytest

from bridge.app.application import BridgeCore
from bridge.app.config import ConfigManager
from bridge.app.events import Topic
from bridge.healthcare.procedures import BUILTIN_PROCEDURES, ProcedureLibrary
from bridge.healthcare.safety import cautions_for_target, check_request
from bridge.voice.assistant import VoiceAssistant, VoiceState, strip_wake_word
from bridge.voice.audio import SAMPLE_RATE, ArraySource, UtteranceCapture, synth_speech_like
from bridge.voice.stt import MockSTT, Transcript, build_stt
from bridge.voice.tts import SilentTTS, build_tts


def _audio(*durations):
    quiet = (np.random.default_rng(1).normal(0, 0.002, SAMPLE_RATE) * 32767).astype(np.int16)
    parts = [quiet]
    for i, d in enumerate(durations):
        parts += [synth_speech_like(d, seed=i), quiet]
    return np.concatenate(parts + [quiet])


def test_vad_segments_two_utterances_and_ignores_clicks():
    audio = _audio(1.0, 0.6)
    click = synth_speech_like(0.05, seed=9)  # 50 ms burst: below min_speech_ms
    audio = np.concatenate([audio, click, np.zeros(SAMPLE_RATE, np.int16)])
    cap = UtteranceCapture()
    src = ArraySource(audio)
    u = [cap.next_utterance(src) for _ in range(3)]
    assert u[0] is not None and 1.5 < u[0].duration_s < 2.5
    assert u[1] is not None and 1.0 < u[1].duration_s < 2.0
    assert u[2] is None
    wav = u[0].wav_bytes()
    assert wav[:4] == b"RIFF" and len(wav) > 1000


@pytest.mark.parametrize("text,wake,expected", [
    ("Hey Bridge, where is the syringe?", "bridge", "where is the syringe?"),
    ("bridge next", "bridge", "next"),
    ("where is it", "bridge", None),
    ("OK Bridge: stop", "bridge", "stop"),
])
def test_strip_wake_word(text, wake, expected):
    assert strip_wake_word(text, wake) == expected


def test_voice_assistant_routes_commands_and_speaks_replies():
    audio = _audio(1.0, 0.8)
    stt = MockSTT(["where is the syringe", "bridge next step"])
    tts = SilentTTS()
    states = []
    va = VoiceAssistant(lambda: ArraySource(audio), stt, tts, lambda t: f"ok: {t}", on_event=lambda e: states.append(e.state))
    va.start()
    deadline = time.time() + 5
    while time.time() < deadline and len(tts.spoken) < 2:
        time.sleep(0.05)
    va.stop()
    assert tts.spoken == ["ok: where is the syringe", "ok: next step"]
    assert VoiceState.HEARD in states and VoiceState.THINKING in states and states[-1] is VoiceState.OFF


def test_voice_assistant_wake_word_required_filters_commands():
    audio = _audio(1.0, 0.8)
    stt = MockSTT(["where is the syringe", "bridge where is the gauze"])
    tts = SilentTTS()
    va = VoiceAssistant(lambda: ArraySource(audio), stt, tts, lambda t: f"ok: {t}", require_wake_word=True)
    va.start()
    deadline = time.time() + 5
    while time.time() < deadline and not tts.spoken:
        time.sleep(0.05)
    va.stop()
    assert tts.spoken == ["ok: where is the gauze"]


def test_low_confidence_or_non_speech_transcripts_are_ignored():
    tr = Transcript(text="", is_speech=False)
    assert tr.command_text == ""
    stt = MockSTT([""])
    tts = SilentTTS()
    va = VoiceAssistant(lambda: ArraySource(_audio(1.0)), stt, tts, lambda t: "should not run")
    va.start()
    time.sleep(0.5)
    va.stop()
    assert tts.spoken == []


def test_stt_and_tts_factories_degrade_gracefully():
    from bridge.ai.base import AIError

    assert isinstance(build_stt("mock", None, "m"), MockSTT)
    with pytest.raises(AIError):
        build_stt("gemini", None, "m")  # no key
    tts = build_tts(enabled=False)
    tts.speak("hello")
    assert tts.spoken == ["hello"]


def test_safety_policy_declines_dosing_and_flags_sharps():
    assert not check_request("How much insulin should I give?").allowed
    assert not check_request("should I inject this now").allowed
    v = check_request("where is the needle")
    assert v.allowed and "sharps" in v.caution_text.lower()
    assert any("label" in c.lower() for c in cautions_for_target("vial", "insulin vial"))
    assert cautions_for_target("gauze") == []


def test_procedure_library_and_fuzzy_lookup():
    lib = ProcedureLibrary(extra_dir=None)
    assert len(lib.procedures) == len(BUILTIN_PROCEDURES) >= 7
    assert lib.find("start the order of draw").id == "order_of_draw"
    assert lib.find("instrument count please").id == "instrument_count"
    assert lib.find("iv tray").id == "iv_setup"
    assert lib.find("make tea") is None
    task = lib.get("order_of_draw").to_task()
    assert len(task.steps) == 7 and task.steps[0].target_object == "blood culture bottle"


@pytest.fixture
def clinic(tmp_path):
    cfg = ConfigManager(tmp_path / "settings.yaml")
    cfg.settings.profiles_dir = tmp_path / "profiles"
    cfg.settings.log_dir = tmp_path / "logs"
    core = BridgeCore(cfg)
    core.enter_simulation(seed=2, scene="clinic")
    core.calibration.method.settle_s = 0.02
    assert core.calibrate().success
    yield core
    core.shutdown()


def test_spoken_healthcare_commands_end_to_end(clinic):
    core = clinic
    steps = []
    core.bus.subscribe(Topic.PROCEDURE_STEP, lambda e: steps.append(e.payload["index"]))
    assert core.handle_spoken("how many mg should I give").startswith("I can locate items")
    r = core.handle_spoken("where is the syringe")
    assert r.startswith("Found syringe")
    r = core.handle_spoken("hey bridge, point to the sharps container")
    assert r.startswith("Pointing to sharps container")
    r = core.handle_spoken("start the order of draw")
    assert "Step 1 of 7" in r and core.guide.active
    r = core.handle_spoken("next")
    assert "Step 2 of 7" in r and "Light blue" in r
    assert core.executor.active is not None  # the light-blue tube is highlighted and tracked
    assert core.handle_spoken("repeat").startswith("Step 2 of 7")
    assert core.handle_spoken("back").startswith("Step 1 of 7")
    assert core.handle_spoken("what next").startswith("Step 2 of 7")
    assert core.handle_spoken("stop procedure") == "Procedure stopped."
    assert not core.guide.active and len(core.renderer.scene) == 0
    assert steps and steps[0] == 0
    assert "Available procedures" in core.handle_spoken("list procedures")


def test_procedure_runs_to_completion(clinic):
    core = clinic
    core.handle_spoken("start ppe donning")
    replies = [core.handle_spoken("next") for _ in range(5)]
    assert replies[-1] == "Procedure complete."
    assert not core.guide.active


def test_registration_trim_shifts_projection_and_persists(clinic):
    core = clinic
    core.save_profile("trim")
    core.handle_spoken("where is the gauze")
    before = core.executor.mapper.camera_to_projector(core.executor.last_states[0].center)
    core.nudge_registration(12, -5)
    after = core.executor.mapper.camera_to_projector(core.executor.last_states[0].center)
    assert abs((after.x - before.x) - 12) < 1e-6 and abs((after.y - before.y) + 5) < 1e-6
    assert core.store.load("trim").trim_px == (12.0, -5.0)
    core.reset_registration_trim()
    assert core.registration_trim == (0.0, 0.0)
