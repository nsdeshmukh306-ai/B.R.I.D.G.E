"""End-to-end pipeline in simulation, device handling, task state machine, config."""
import time

import numpy as np
import pytest

from bridge.app.application import BridgeCore
from bridge.app.config import AppSettings, ConfigManager, Secrets
from bridge.app.events import EventBus, Topic
from bridge.app.state import CalibrationStatus
from bridge.camera.device import CameraDevice, CameraInfo
from bridge.camera.manager import CameraManager
from bridge.interaction.commands import ClearProjection, HighlightObject, TargetSpec
from bridge.interaction.executor import CommandExecutor
from bridge.interaction.tasks import Task, TaskRunner, TaskState, TaskStep
from bridge.projector.manager import DisplayManager
from bridge.render.renderer import ProjectionRenderer
from bridge.spatial.homography import apply_homography


@pytest.fixture
def core(tmp_path):
    cfg = ConfigManager(tmp_path / "settings.yaml")
    cfg.settings.profiles_dir = tmp_path / "profiles"
    cfg.settings.log_dir = tmp_path / "logs"
    c = BridgeCore(cfg)
    c.enter_simulation(seed=2)
    c.calibration.method.settle_s = 0.02
    yield c
    c.shutdown()


def test_simulation_starts_without_hardware(core):
    assert core.state.mode == "simulation"
    assert core.cameras.wait_for_frame(3.0) is not None
    assert core.state.diagnostics.camera_connected


def test_spatial_command_refused_without_calibration(core):
    core.cameras.wait_for_frame(3.0)
    r = core.ask("Where is the screwdriver?")
    assert not r.ok and "calibration required" in r.message.lower()


def test_full_demo_calibrate_ask_follow(core):
    res = core.calibrate()
    assert res.success and core.state.is_ready and core.state.calibration_status is CalibrationStatus.VALID
    world, proj, cam = core.simulation
    r = core.ask("Where is the screwdriver?")
    assert r.ok and r.n_targets == 1
    # The projected circle must land on the physical object (table space).
    tbl = apply_homography(proj.H_proj_to_table, [r.projector_points[0]])[0]
    o = world.get("obj-screwdriver")
    assert np.hypot(tbl[0] - o.x, tbl[1] - o.y) < 15
    # Move the object; the projection must follow.
    for _ in range(20):
        world.move_by("obj-screwdriver", -6, 4)
        time.sleep(0.06)
    s = core.executor.last_states[0]
    pp = core.executor.mapper.camera_to_projector(s.center)
    tbl = apply_homography(proj.H_proj_to_table, [pp])[0]
    o = world.get("obj-screwdriver")
    assert np.hypot(tbl[0] - o.x, tbl[1] - o.y) < 30
    assert core.state.diagnostics.tracking_target == "screwdriver"


def test_second_demo_commands(core):
    assert core.calibrate().success
    assert core.ask("Highlight the screwdriver").n_targets == 1
    r = core.ask("Where are the screws?")
    assert r.ok and r.n_targets == 2
    r = core.ask("Point to the scissors")
    assert r.ok and r.strategy == "ai_boxes"
    r = core.ask("Where is the hammer?")
    assert r.strategy == "message" and "uncertain" in r.message.lower()
    r = core.ask("Move the pen to the phone")
    assert r.ok and r.strategy == "path"
    assert core.ask("clear").strategy == "clear" and len(core.renderer.scene) == 0
    assert core.next_step().ok


def test_target_lost_clears_projection(core):
    assert core.calibrate().success
    world, proj, cam = core.simulation
    events = []
    core.bus.subscribe(Topic.TRACKING_LOST, lambda e: events.append(e))
    assert core.ask("Where is the phone?").ok
    world.remove("obj-phone")
    deadline = time.time() + 5
    while time.time() < deadline and not events:
        time.sleep(0.05)
    assert events, "TRACKING_LOST was not published"
    assert core.executor.active is None
    assert all(p.group != "target" for p in core.renderer.scene.items())


def test_profile_persists_and_reloads(core):
    assert core.calibrate().success
    assert core.save_profile("test")
    core.invalidate_calibration("test")
    assert not core.state.is_ready
    assert core.try_load_profile() and core.state.is_ready


def test_display_change_invalidates_calibration(core):
    from bridge.projector.manager import DisplayDevice

    assert core.calibrate().success
    core.select_display(DisplayDevice(id="display:9", name="Other", width=1920, height=1080, x=0, y=0, is_primary=False))
    assert core.state.calibration_status is CalibrationStatus.NOT_CALIBRATED


def test_missing_camera_does_not_crash():
    bus = EventBus()
    msgs = []
    bus.subscribe(Topic.STATUS_MESSAGE, lambda e: msgs.append(e.payload["text"]))
    cm = CameraManager(bus)
    info = CameraInfo(id="cam:99", name="Ghost", index=99)
    assert cm.open(info) is False and not cm.is_open and cm.latest_frame() is None
    assert msgs and "unavailable" in msgs[0].lower()
    dev = CameraDevice(info)
    assert dev.read_frame() is None
    dev.release()
    cm.close()


def test_display_enumeration_does_not_crash():
    dm = DisplayManager()
    dm.refresh()  # may be empty in CI
    assert dm.select("nonexistent") is None


def test_executor_without_mapper_shows_calibration_message():
    ex = CommandExecutor(ProjectionRenderer(320, 240))
    r = ex.execute(HighlightObject(target=TargetSpec(label="x")), np.zeros((240, 320, 3), np.uint8))
    assert not r.ok and any(p.kind == "message" for p in ex.renderer.scene.items())
    assert ex.execute(ClearProjection(), np.zeros((240, 320, 3), np.uint8)).ok


def test_task_state_machine():
    task = Task(id="t1", name="demo", steps=[TaskStep(instruction="Pick up screwdriver", target_object="screwdriver"),
                                             TaskStep(instruction="Pick up screw", target_object="screw")])
    seen = []
    runner = TaskRunner(task, on_state=lambda s, step: seen.append(s))
    step = runner.start()
    assert step.target_object == "screwdriver" and runner.state is TaskState.UNDERSTAND
    runner.located()
    assert runner.state is TaskState.WAIT_FOR_ACTION
    assert runner.action_observed(False).target_object == "screwdriver"
    assert runner.action_observed(True).target_object == "screw"
    runner.located()
    assert runner.action_observed(True) is None and runner.state is TaskState.DONE
    with pytest.raises(ValueError):
        runner.transition(TaskState.PROJECT)


def test_config_load_save_and_env_secrets(tmp_path, monkeypatch):
    cfg = ConfigManager(tmp_path / "s.yaml")
    cfg.settings.ai.model = "gemini-x"
    cfg.save()
    assert ConfigManager(tmp_path / "s.yaml").settings.ai.model == "gemini-x"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert Secrets.from_env(None).gemini_api_key is None
    (tmp_path / ".env").write_text("GEMINI_API_KEY=abc123\n")
    assert Secrets.from_env(tmp_path / ".env").gemini_api_key == "abc123"
    assert AppSettings().render.background == "black"


def test_eventbus_isolates_bad_subscribers():
    bus = EventBus()
    got = []
    bus.subscribe(Topic.STATUS_MESSAGE, lambda e: 1 / 0)
    unsub = bus.subscribe(Topic.STATUS_MESSAGE, lambda e: got.append(e.payload["text"]))
    bus.publish(Topic.STATUS_MESSAGE, text="hi")
    unsub()
    bus.publish(Topic.STATUS_MESSAGE, text="again")
    assert got == ["hi"]
