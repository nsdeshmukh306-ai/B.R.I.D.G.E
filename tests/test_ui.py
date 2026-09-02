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
    assert "2.8 px" in wiz.lbl_result.text() and wiz.btn_save.isEnabled()
    wiz._on_done(CalibrationResult(success=False, message="Calibration failed.\nPossible causes: ..."))
    assert not wiz.btn_save.isEnabled()
