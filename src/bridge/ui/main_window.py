"""BRIDGE main control window."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSplitter, QVBoxLayout, QWidget,
)

from bridge.app.application import BridgeCore
from bridge.app.diagnostics import diagnostics_report
from bridge.app.events import Topic
from bridge.app.state import CalibrationStatus
from bridge.interaction.commands import ClearProjection
from bridge.projector.manager import DisplayDevice
from bridge.projector.window import ProjectionWindow
from bridge.spatial.geometry import Point
from bridge.ui.calibration_wizard import CalibrationWizard
from bridge.ui.settings import SettingsDialog
from bridge.ui.widgets import ImageView, LogPanel, run_in_background

log = logging.getLogger("bridge.ui")

WELCOME = ("WELCOME TO BRIDGE\n\nConnect:\n  1. Camera\n  2. Projector\n\nThen press SET UP BRIDGE — or switch to "
           "Simulation mode to try everything without hardware.")
READY = ('BRIDGE READY\n\nYour physical environment is now an AI interface.\n\nTry:\n  "Where is the screwdriver?"\n'
         '  "Highlight the red object."\n  "Point to the scissors."\n  "Where are the screws?"')


class _WindowProjectorLink:
    """Adapts the ProjectionWindow to the calibration engine's ProjectorLink protocol."""

    def __init__(self, window: ProjectionWindow):
        self.window = window

    def show_image(self, image: np.ndarray) -> None:
        # Marshal to the GUI thread; calibration runs on a worker.
        from PySide6.QtCore import QMetaObject, Q_ARG, Qt as _Qt  # noqa: N814

        is_white = image.mean() > 250
        QMetaObject.invokeMethod(self.window, "set_override_slot", _Qt.ConnectionType.BlockingQueuedConnection,
                                 Q_ARG(object, None if is_white else image))

    def size(self) -> tuple[int, int]:
        return (self.window.renderer.width, self.window.renderer.height)


class MainWindow(QMainWindow):
    status_signal = Signal(str, str)

    def __init__(self, core: BridgeCore):
        super().__init__()
        self.core = core
        self.setWindowTitle("BRIDGE — Universal Spatial AI")
        self.resize(1320, 820)
        self.projection: Optional[ProjectionWindow] = None
        self._drag_obj: Optional[str] = None
        self._build()
        self._wire_events()
        self.status_signal.connect(self._on_status)
        self._ui_timer = QTimer(self)
        self._ui_timer.setInterval(66)
        self._ui_timer.timeout.connect(self._refresh)
        self._ui_timer.start()
        self.refresh_devices()
        if core.settings.mode == "simulation":
            self.rb_sim.setChecked(True)
            self._on_mode_changed()

    # ---- layout ------------------------------------------------------------------------------
    def _build(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root = QHBoxLayout(central)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        root.addWidget(splitter)

        # left: control panel (scrollable)
        panel = QWidget()
        pl = QVBoxLayout(panel)
        pl.setSpacing(10)
        title = QLabel("BRIDGE")
        title.setObjectName("title")
        sub = QLabel("Universal Spatial AI")
        sub.setObjectName("subtitle")
        pl.addWidget(title)
        pl.addWidget(sub)

        g = QGroupBox("MODE")
        gl = QHBoxLayout(g)
        self.rb_phys = QRadioButton("Physical")
        self.rb_sim = QRadioButton("Simulation")
        self.rb_phys.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.rb_phys)
        grp.addButton(self.rb_sim)
        self.rb_phys.toggled.connect(self._on_mode_changed)
        gl.addWidget(self.rb_phys)
        gl.addWidget(self.rb_sim)
        pl.addWidget(g)

        g = QGroupBox("CAMERA")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        self.cb_camera = QComboBox()
        self.cb_camera.currentIndexChanged.connect(self._on_camera_selected)
        self.btn_refresh = QPushButton("⟳")
        self.btn_refresh.setFixedWidth(36)
        self.btn_refresh.clicked.connect(self.refresh_devices)
        row.addWidget(self.cb_camera, 1)
        row.addWidget(self.btn_refresh)
        gl.addLayout(row)
        self.lbl_cam_status = QLabel("Status: Not connected")
        self.lbl_cam_status.setObjectName("status_warn")
        gl.addWidget(self.lbl_cam_status)
        pl.addWidget(g)

        g = QGroupBox("DISPLAY / PROJECTOR")
        gl = QVBoxLayout(g)
        self.cb_display = QComboBox()
        self.cb_display.currentIndexChanged.connect(self._on_display_selected)
        gl.addWidget(self.cb_display)
        self.lbl_disp_status = QLabel("Status: Not connected")
        self.lbl_disp_status.setObjectName("status_warn")
        gl.addWidget(self.lbl_disp_status)
        row = QHBoxLayout()
        self.btn_open_proj = QPushButton("OPEN PROJECTION")
        self.btn_open_proj.setObjectName("accent")
        self.btn_open_proj.clicked.connect(self.open_projection)
        self.btn_close_proj = QPushButton("CLOSE")
        self.btn_close_proj.clicked.connect(self.close_projection)
        self.btn_close_proj.setEnabled(False)
        row.addWidget(self.btn_open_proj)
        row.addWidget(self.btn_close_proj)
        gl.addLayout(row)
        self.btn_test = QPushButton("Test graphics")
        self.btn_test.clicked.connect(self._test_graphics)
        gl.addWidget(self.btn_test)
        pl.addWidget(g)

        g = QGroupBox("SURFACE")
        gl = QHBoxLayout(g)
        self.surface_group = QButtonGroup(self)
        for i, name in enumerate(("Table", "Wall", "Custom")):
            b = QPushButton(name)
            b.setCheckable(True)
            b.setChecked(i == 0)
            self.surface_group.addButton(b, i)
            gl.addWidget(b)
        self.surface_group.idClicked.connect(self._on_surface)
        pl.addWidget(g)

        g = QGroupBox("SPATIAL CALIBRATION")
        gl = QVBoxLayout(g)
        self.lbl_cal_status = QLabel("Status: Not calibrated")
        self.lbl_cal_status.setObjectName("status_warn")
        gl.addWidget(self.lbl_cal_status)
        self.btn_calibrate = QPushButton("AUTO CALIBRATE")
        self.btn_calibrate.setObjectName("primary")
        self.btn_calibrate.clicked.connect(self.start_calibration)
        gl.addWidget(self.btn_calibrate)
        row = QHBoxLayout()
        self.btn_load_profile = QPushButton("Load profile")
        self.btn_load_profile.clicked.connect(self._load_profile)
        self.btn_validate = QPushButton("Click-to-project test")
        self.btn_validate.setCheckable(True)
        self.btn_validate.setToolTip("Click in the camera view to project a marker at that physical spot.")
        row.addWidget(self.btn_load_profile)
        row.addWidget(self.btn_validate)
        gl.addLayout(row)
        pl.addWidget(g)

        g = QGroupBox("AI COMMAND")
        gl = QVBoxLayout(g)
        self.edit_cmd = QLineEdit()
        self.edit_cmd.setPlaceholderText("Where is the screwdriver?")
        self.edit_cmd.returnPressed.connect(self.execute_command)
        gl.addWidget(self.edit_cmd)
        row = QHBoxLayout()
        self.btn_execute = QPushButton("EXECUTE")
        self.btn_execute.setObjectName("primary")
        self.btn_execute.clicked.connect(self.execute_command)
        self.btn_next = QPushButton("What next?")
        self.btn_next.setToolTip("Assembly assistance: ask the AI for the next step")
        self.btn_next.clicked.connect(self.next_step)
        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_projection)
        row.addWidget(self.btn_execute, 2)
        row.addWidget(self.btn_next)
        row.addWidget(self.btn_clear)
        gl.addLayout(row)
        self.lbl_ai_result = QLabel("")
        self.lbl_ai_result.setWordWrap(True)
        self.lbl_ai_result.setObjectName("subtitle")
        gl.addWidget(self.lbl_ai_result)
        pl.addWidget(g)

        row = QHBoxLayout()
        self.btn_settings = QPushButton("Settings")
        self.btn_settings.clicked.connect(self.open_settings)
        row.addWidget(self.btn_settings)
        pl.addLayout(row)
        pl.addStretch()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(panel)
        scroll.setMinimumWidth(360)
        scroll.setMaximumWidth(420)
        splitter.addWidget(scroll)

        # centre: views
        centre = QWidget()
        cl = QVBoxLayout(centre)
        self.lbl_banner = QLabel(WELCOME)
        self.lbl_banner.setObjectName("mono")
        self.lbl_banner.setStyleSheet("background:#1b2028; border-radius:8px; padding:12px; font-family: Menlo, Consolas, monospace;")
        cl.addWidget(self.lbl_banner)
        views = QHBoxLayout()
        vb = QVBoxLayout()
        vb.addWidget(QLabel("Camera view"))
        self.view_camera = ImageView("Camera view")
        self.view_camera.clicked.connect(self._on_camera_click)
        self.view_camera.dragged.connect(self._on_camera_drag)
        self.view_camera.released.connect(lambda _p: setattr(self, "_drag_obj", None))
        vb.addWidget(self.view_camera, 1)
        views.addLayout(vb, 1)
        vb = QVBoxLayout()
        vb.addWidget(QLabel("Projection output"))
        self.view_projection = ImageView("Projection output")
        vb.addWidget(self.view_projection, 1)
        views.addLayout(vb, 1)
        cl.addLayout(views, 3)
        self.lbl_sim_hint = QLabel("Simulation: drag objects in the camera view to move them on the virtual table.")
        self.lbl_sim_hint.setObjectName("subtitle")
        self.lbl_sim_hint.hide()
        cl.addWidget(self.lbl_sim_hint)
        splitter.addWidget(centre)

        # right: diagnostics + log
        right = QWidget()
        rl = QVBoxLayout(right)
        g = QGroupBox("DIAGNOSTICS")
        gl = QVBoxLayout(g)
        self.lbl_diag = QLabel("")
        self.lbl_diag.setObjectName("mono")
        self.lbl_diag.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        gl.addWidget(self.lbl_diag)
        rl.addWidget(g)
        g = QGroupBox("LOG")
        gl = QVBoxLayout(g)
        self.log_panel = LogPanel()
        logging.getLogger("bridge").addHandler(self.log_panel)
        gl.addWidget(self.log_panel)
        rl.addWidget(g, 1)
        right.setMinimumWidth(280)
        right.setMaximumWidth(360)
        splitter.addWidget(right)
        splitter.setSizes([380, 700, 300])
        self.statusBar().showMessage("Ready")

    # ---- events from core --------------------------------------------------------------------------
    def _wire_events(self) -> None:
        bus = self.core.bus
        bus.subscribe(Topic.STATUS_MESSAGE, lambda e: self.status_signal.emit(e.payload.get("text", ""), e.payload.get("level", "info")))
        bus.subscribe(Topic.CALIBRATION_INVALIDATED, lambda e: self.status_signal.emit(e.payload.get("reason", ""), "warning"))
        bus.subscribe(Topic.TRACKING_LOST, lambda e: self.status_signal.emit("Target lost.", "warning"))

    def _on_status(self, text: str, level: str) -> None:
        self.statusBar().showMessage(text, 8000)
        if level in ("warning", "error"):
            log.warning(text) if level == "warning" else log.error(text)

    # ---- devices ----------------------------------------------------------------------------------
    def refresh_devices(self) -> None:
        cams, disps = self.core.refresh_devices()
        self.cb_camera.blockSignals(True)
        self.cb_camera.clear()
        if self.core.state.mode == "simulation":
            self.cb_camera.addItem("Simulated camera", None)
        elif cams:
            for c in cams:
                self.cb_camera.addItem(f"{c.name}  ({c.width}x{c.height})", c)
        else:
            self.cb_camera.addItem("No cameras found", None)
        self.cb_camera.blockSignals(False)
        self.cb_display.blockSignals(True)
        self.cb_display.clear()
        if self.core.state.mode == "simulation":
            self.cb_display.addItem("Simulated projector (1280x720)", None)
        else:
            for d in disps:
                self.cb_display.addItem(d.label, d)
            if not disps:
                self.cb_display.addItem("No displays found", None)
        self.cb_display.blockSignals(False)
        if self.core.state.mode == "physical":
            if cams:
                self._on_camera_selected(0)
            sug = self.core.displays.suggest_projector()
            if sug is not None:
                for i in range(self.cb_display.count()):
                    if self.cb_display.itemData(i) is sug:
                        self.cb_display.setCurrentIndex(i)
                        break
                self._on_display_selected(self.cb_display.currentIndex())

    def _on_camera_selected(self, idx: int) -> None:
        info = self.cb_camera.itemData(idx) if idx >= 0 else None
        if info is None or self.core.state.mode == "simulation":
            return
        self.lbl_cam_status.setText("Status: Connecting...")
        run_in_background(lambda: self.core.select_camera(info), self._camera_opened)

    def _camera_opened(self, ok: bool) -> None:
        self._set_status(self.lbl_cam_status, "Status: Connected" if ok else "Status: Unavailable", ok)
        if ok:
            self.core.try_load_profile()
            self._update_banner()

    def _on_display_selected(self, idx: int) -> None:
        d: Optional[DisplayDevice] = self.cb_display.itemData(idx) if idx >= 0 else None
        if d is None:
            return
        self.core.select_display(d)
        self._set_status(self.lbl_disp_status, f"Status: Selected ({d.width}x{d.height})", True)
        if self.projection is not None and self.projection.isVisible():
            self.projection.show_on(d)
        self.core.try_load_profile()

    def _on_surface(self, idx: int) -> None:
        self.core.state.surface_type = ("table", "wall", "custom")[idx]
        log.info("Surface preset: %s", self.core.state.surface_type)

    # ---- projection window ---------------------------------------------------------------------------
    def open_projection(self) -> None:
        if self.core.state.mode == "simulation":
            self.statusBar().showMessage("Simulation mode: projection output is shown in the right-hand view.", 5000)
            return
        d = self.core.projector_display
        if d is None:
            QMessageBox.warning(self, "BRIDGE", "Select a display first.")
            return
        if self.projection is None:
            self.projection = ProjectionWindow(self.core.renderer, self.core.settings.render.fps)
            self.projection.frame_listeners.append(self._on_projection_frame)
        self.projection.show_on(d)
        self.core.attach_projector(_WindowProjectorLink(self.projection))
        self.btn_close_proj.setEnabled(True)
        self._set_status(self.lbl_disp_status, f"Status: Projecting on {d.name}", True)

    def close_projection(self) -> None:
        if self.projection is not None:
            self.projection.close_projection()
        self.btn_close_proj.setEnabled(False)
        self.core.projector_link = None
        self.core.invalidate_calibration("Projection window closed. Calibration invalidated.")
        self._set_status(self.lbl_disp_status, "Status: Selected", True)

    def _on_projection_frame(self, frame: np.ndarray) -> None:
        self._last_proj_frame = frame

    def _test_graphics(self) -> None:
        r = self.core.renderer
        r.clear()
        w, h = r.width, r.height
        r.circle(Point(x=w * 0.3, y=h * 0.5), min(w, h) * 0.12)
        r.arrow(Point(x=w * 0.55, y=h * 0.25), Point(x=w * 0.7, y=h * 0.45))
        r.label(Point(x=w * 0.55, y=h * 0.2), "BRIDGE test")
        r.target_zone([Point(x=w * 0.6, y=h * 0.6), Point(x=w * 0.85, y=h * 0.6), Point(x=w * 0.85, y=h * 0.85), Point(x=w * 0.6, y=h * 0.85)])
        r.path([Point(x=w * 0.1, y=h * 0.85), Point(x=w * 0.3, y=h * 0.75), Point(x=w * 0.5, y=h * 0.9)])
        log.info("Test graphics drawn at %dx%d", w, h)

    # ---- calibration --------------------------------------------------------------------------------
    def start_calibration(self) -> None:
        ok, why = self.core.can_calibrate()
        if not ok:
            QMessageBox.information(self, "BRIDGE", why)
            return
        wiz = CalibrationWizard(self.core.calibrate, lambda: self.core.save_profile("default"),
                                self.core.state.diagnostics.camera_name, self.core.state.diagnostics.display_name, self)
        wiz.saved.connect(self._update_banner)
        wiz.exec()
        self._update_banner()

    def _load_profile(self) -> None:
        if self.core.try_load_profile():
            self.statusBar().showMessage("Profile loaded.", 4000)
        else:
            self.statusBar().showMessage("No valid profile matches the current camera/display.", 5000)
        self._update_banner()

    def _on_camera_click(self, p: Point) -> None:
        if self.core.state.mode == "simulation" and self.core.simulation is not None and not self.btn_validate.isChecked():
            world, _proj, cam = self.core.simulation
            obj = world.object_at(cam.camera_to_table(p))
            self._drag_obj = obj.id if obj else None
            return
        if self.btn_validate.isChecked() and self.core.executor.mapper is not None:
            pp = self.core.executor.mapper.camera_to_projector(p)
            self.core.renderer.scene.remove_group("probe")
            self.core.renderer.point(pp, group="probe")
            self.core.renderer.circle(pp, 40, group="probe")
            log.info("Probe: camera (%.0f,%.0f) -> projector (%.0f,%.0f)", p.x, p.y, pp.x, pp.y)

    def _on_camera_drag(self, p: Point) -> None:
        if self._drag_obj and self.core.simulation is not None:
            world, _proj, cam = self.core.simulation
            t = cam.camera_to_table(p)
            world.move(self._drag_obj, t.x, t.y)

    # ---- AI ------------------------------------------------------------------------------------------
    def execute_command(self) -> None:
        q = self.edit_cmd.text().strip()
        if not q:
            return
        if not self.core.state.is_ready:
            self.lbl_ai_result.setText("Spatial calibration required.")
            self.statusBar().showMessage("Spatial calibration required.", 5000)
            return
        self.btn_execute.setEnabled(False)
        self.lbl_ai_result.setText("Thinking...")
        run_in_background(lambda: self.core.ask(q), self._on_ai_done, lambda e: self._on_ai_done(None, e))

    def next_step(self) -> None:
        if not self.core.state.is_ready:
            self.statusBar().showMessage("Spatial calibration required.", 5000)
            return
        self.btn_execute.setEnabled(False)
        self.lbl_ai_result.setText("Planning next step...")
        run_in_background(lambda: self.core.next_step(), self._on_ai_done, lambda e: self._on_ai_done(None, e))

    def _on_ai_done(self, result, error: str | None = None) -> None:
        self.btn_execute.setEnabled(True)
        if error or result is None:
            self.lbl_ai_result.setText(f"Error: {error}")
            return
        ident = self.core.last_identification
        detail = f" — intent={ident.intent}, confidence={ident.confidence:.2f}" if ident else ""
        self.lbl_ai_result.setText(("✓ " if result.ok else "✗ ") + result.message + detail)

    def clear_projection(self) -> None:
        self.core.execute(ClearProjection()) if self.core.cameras.is_open else self.core.executor.clear()

    # ---- mode ----------------------------------------------------------------------------------------
    def _on_mode_changed(self) -> None:
        sim = self.rb_sim.isChecked()
        if sim and self.core.simulation is None:
            self.close_projection() if self.projection else None
            self.core.enter_simulation()
            self.lbl_sim_hint.show()
            self._set_status(self.lbl_cam_status, "Status: Simulated camera", True)
            self._set_status(self.lbl_disp_status, "Status: Simulated projector", True)
            self.refresh_devices()
        elif not sim and self.core.simulation is not None:
            self.core.leave_simulation()
            self.lbl_sim_hint.hide()
            self.view_camera.set_frame(None)
            self.view_projection.set_frame(None)
            self._set_status(self.lbl_cam_status, "Status: Not connected", False)
            self.refresh_devices()
        self._update_banner()

    # ---- periodic refresh -----------------------------------------------------------------------------
    def _refresh(self) -> None:
        frame = self.core.cameras.latest_frame()
        if frame is not None:
            self.view_camera.set_frame(frame)
        if self.core.state.mode == "simulation" and self.core.simulation is not None:
            self.view_projection.set_frame(self.core.simulation[1].current_image())
        elif self.projection is not None and self.projection.isVisible():
            self.view_projection.set_frame(self.projection.current_frame())
        st = self.core.state
        cal = st.calibration_status
        if cal is CalibrationStatus.VALID:
            err = st.diagnostics.calibration_mean_error
            self._set_status(self.lbl_cal_status, f"Status: VALID ({err:.1f} px)" if err is not None else "Status: VALID", True)
        elif cal is CalibrationStatus.CALIBRATING:
            self.lbl_cal_status.setText("Status: Calibrating...")
        else:
            self._set_status(self.lbl_cal_status, f"Status: {cal.value}", False)
        self.lbl_diag.setText(diagnostics_report(st))

    def _set_status(self, label: QLabel, text: str, ok: bool) -> None:
        label.setText(text)
        label.setObjectName("status_ok" if ok else "status_warn")
        label.style().unpolish(label)
        label.style().polish(label)

    def _update_banner(self) -> None:
        self.lbl_banner.setText(READY if self.core.state.is_ready else WELCOME)

    def open_settings(self) -> None:
        dlg = SettingsDialog(self.core.config, self)
        if dlg.exec():
            self.statusBar().showMessage("Settings saved. Restart BRIDGE to apply hardware/calibration changes.", 6000)
            r = self.core.settings.render
            self.core.renderer.background = r.background if self.core.state.mode == "physical" else "white"
            self.core.executor.line_width = r.line_width

    def closeEvent(self, event) -> None:  # noqa: N802
        self._ui_timer.stop()
        logging.getLogger("bridge").removeHandler(self.log_panel)
        if self.projection is not None:
            self.projection.close_projection()
        self.core.shutdown()
        super().closeEvent(event)
