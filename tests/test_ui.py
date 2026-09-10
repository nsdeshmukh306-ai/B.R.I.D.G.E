"""GUI smoke tests (offscreen Qt)."""
import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def test_main_window_opens_and_enters_simulation(qapp, tmp_path):
    from bridge.app.application import BridgeCore
    from bridge.app.config import ConfigManager
    from bridge.ui.main_window import MainWindow

    cfg = ConfigManager(tmp_path / "s.yaml")
    cfg.settings.mode = "simulation"
    cfg.settings.profiles_dir = tmp_path / "p"
    cfg.settings.log_dir = tmp_path / "l"
    core = BridgeCore(cfg)
    win = MainWindow(core)
    win.show()
    qapp.processEvents()
    assert core.simulation is not None
    assert "Simulated" in win.cb_camera.currentText()
    win._test_graphics()
    assert len(core.renderer.scene) == 5
    win.close()


def test_surgical_panel_drives_a_case_and_shows_the_count_board(qapp, tmp_path):
    """Every button on the panel goes through the same spoken entry point as the voice."""
    from bridge.app.application import BridgeCore
    from bridge.app.config import ConfigManager
    from bridge.surgical.monitor import Alert
    from bridge.ui.main_window import MainWindow

    cfg = ConfigManager(tmp_path / "s2.yaml")
    cfg.settings.mode = "simulation"
    cfg.settings.profiles_dir = tmp_path / "p2"
    cfg.settings.log_dir = tmp_path / "l2"
    cfg.settings.assistant.records_dir = tmp_path / "records"
    core = BridgeCore(cfg)
    win = MainWindow(core)
    win.show()
    qapp.processEvents()
    panel = win.surgical
    assert panel.isEnabled() and panel.cb_set.count() > 0

    panel.cb_set.setCurrentIndex(panel.cb_set.findData("minor"))
    panel._start_case()
    qapp.processEvents()
    assert core.assistant.case.active

    panel._say("start the count")
    panel._say("count as per the sheet")
    panel.refresh(force=True)
    qapp.processEvents()
    assert panel.table.rowCount() == len(core.assistant.counts.lines)
    assert "reconciled" in panel.lbl_phase.text()

    panel.on_alert(Alert(key="k", level="critical", text="Count discrepancy."))
    qapp.processEvents()
    assert "Count discrepancy." in panel.lbl_alerts.text()

    panel.chk_board.setChecked(False)
    qapp.processEvents()
    assert core.assistant.show_board is False
    win.close()
    core.shutdown()


def test_projection_window_renders_offscreen(qapp):
    from bridge.projector.window import ProjectionWindow
    from bridge.render.renderer import ProjectionRenderer
    from bridge.spatial.geometry import Point

    r = ProjectionRenderer(320, 180)
    r.circle(Point(x=160, y=90), 40)
    w = ProjectionWindow(r, fps=30)
    w.show_windowed(320, 180)
    qapp.processEvents()
    frame = w.current_frame()
    assert frame.shape == (180, 320, 3)
    w.set_override(None)
    w.close_projection()


def test_calibration_wizard_flow(qapp):
    from bridge.spatial.calibration import CalibrationResult
    from bridge.spatial.validation import ValidationResult
    from bridge.ui.calibration_wizard import CalibrationWizard

    val = ValidationResult(mean_error_px=2.8, max_error_px=6.1, median_error_px=2.5, n_points=9, threshold_px=10, valid=True)
    wiz = CalibrationWizard(lambda progress: CalibrationResult(success=True, validation=val), lambda: True, "cam", "disp")
    wiz._on_done(CalibrationResult(success=True, validation=val))
    assert "2.8 camera px" in wiz.lbl_result.text() and wiz.btn_save.isEnabled()
    wiz._on_done(CalibrationResult(success=False, message="Calibration failed.\nPossible causes: ..."))
    assert not wiz.btn_save.isEnabled()


def test_projection_window_accepts_images_from_worker_thread(qapp):
    """Physical-mode calibration path: a worker thread pushes pattern images into the window."""
    import threading
    import time

    from PySide6.QtCore import QTimer
    from bridge.projector.window import ProjectionWindow
    from bridge.render.renderer import ProjectionRenderer, calibration_pattern, solid
    from bridge.spatial.geometry import Point
    from bridge.ui.main_window import _WindowProjectorLink

    w = ProjectionWindow(ProjectionRenderer(320, 180), fps=30)
    w.show_windowed(320, 180)
    link = _WindowProjectorLink(w)
    seen = []
    errors = []

    def worker():
        try:
            link.show_image(calibration_pattern(320, 180, [Point(x=40, y=40)], 10))
            seen.append(w.current_frame()[40, 40].tolist())
            link.show_image(solid(320, 180, (255, 255, 255)))
            seen.append(w._override)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    t = threading.Thread(target=worker)
    QTimer.singleShot(50, t.start)
    for _ in range(200):
        qapp.processEvents()
        if not t.is_alive() and (seen or errors):
            break
        time.sleep(0.01)
    t.join(timeout=2)
    assert not errors, errors
    assert seen[0] == [255, 255, 255] and seen[1] is None
    w.close_projection()
