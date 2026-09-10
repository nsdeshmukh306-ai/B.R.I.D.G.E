"""BRIDGE main control window."""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from PySide6.QtCore import QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QButtonGroup, QCheckBox, QComboBox, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
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
from bridge.ui.surgical_panel import SurgicalPanel
from bridge.ui.widgets import ImageView, LogPanel, run_in_background

log = logging.getLogger("bridge.ui")

WELCOME = ("WELCOME TO BRIDGE\n\nConnect:\n  1. Camera\n  2. Projector\n\nThen OPEN PROJECTION and AUTO CALIBRATE — or switch to "
           "Simulation mode to try everything without hardware.")
READY = ('BRIDGE READY — say it, or type it\n\nTry:\n  "Where is the syringe?"       "Point to the sharps container."\n'
         '  "Highlight the lavender tube."   "Where should I put the needle?"\n'
         '  "Start the order of draw."      then "next", "repeat", "back", "stop".\n\n'
         'BRIDGE locates items and guides protocol steps. It does not diagnose, prescribe or decide doses.')


class _WindowProjectorLink:
    """Adapts the ProjectionWindow to the calibration engine's ProjectorLink protocol."""

    def __init__(self, window: ProjectionWindow):
        self.window = window

    def show_image(self, image: np.ndarray) -> None:
        # Calibration runs on a worker thread; the window marshals this to the GUI thread.
        is_white = image.mean() > 250  # the "back to white canvas" frame: resume normal scene rendering
        self.window.show_image_threadsafe(None if is_white else image)

    def size(self) -> tuple[int, int]:
        return (self.window.renderer.width, self.window.renderer.height)


class MainWindow(QMainWindow):
    status_signal = Signal(str, str)
    voice_signal = Signal(object)
    step_signal = Signal(object, int, int)

    def __init__(self, core: BridgeCore):
        super().__init__()
        self.core = core
        self.setWindowTitle("BRIDGE — Spatial AI for Healthcare")
        self.resize(1320, 820)
        self.projection: Optional[ProjectionWindow] = None
        self._drag_obj: Optional[str] = None
        self._build()
        self._wire_events()
        self.status_signal.connect(self._on_status)
        self.voice_signal.connect(self._on_voice_event)
        self.step_signal.connect(self._on_step)
        self._ptt_capture = None
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
        sub = QLabel("Spatial AI assistant for healthcare workspaces")
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
        self.lbl_trim = QLabel("Registration trim: 0, 0 px  (test mode: arrow keys nudge, Shift = ×10, R = reset)")
        self.lbl_trim.setObjectName("subtitle")
        self.lbl_trim.setWordWrap(True)
        gl.addWidget(self.lbl_trim)
        pl.addWidget(g)

        g = QGroupBox("AI COMMAND")
        gl = QVBoxLayout(g)
        self.edit_cmd = QLineEdit()
        self.edit_cmd.setPlaceholderText("Where is the syringe?")
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

        g = QGroupBox("VOICE")
        gl = QVBoxLayout(g)
        row = QHBoxLayout()
        self.btn_listen = QPushButton("🎙  LISTEN")
        self.btn_listen.setObjectName("accent")
        self.btn_listen.setCheckable(True)
        self.btn_listen.toggled.connect(self._toggle_listen)
        self.btn_ptt = QPushButton("Push to talk")
        self.btn_ptt.setToolTip("Hold to capture one command (or use the Space bar while the window has focus)")
        self.btn_ptt.pressed.connect(self._ptt_pressed)
        self.btn_ptt.released.connect(self._ptt_released)
        row.addWidget(self.btn_listen, 2)
        row.addWidget(self.btn_ptt, 1)
        gl.addLayout(row)
        row = QHBoxLayout()
        self.chk_wake = QCheckBox("Require wake word")
        self.chk_wake.setChecked(self.core.settings.voice.require_wake_word)
        self.chk_wake.toggled.connect(self._on_wake_toggle)
        self.chk_speak = QCheckBox("Speak replies")
        self.chk_speak.setChecked(self.core.settings.voice.tts_enabled)
        self.chk_speak.toggled.connect(self._on_speak_toggle)
        row.addWidget(self.chk_wake)
        row.addWidget(self.chk_speak)
        gl.addLayout(row)
        self.lbl_voice_state = QLabel("Voice: off  (uses this computer's microphone)")
        self.lbl_voice_state.setObjectName("subtitle")
        gl.addWidget(self.lbl_voice_state)
        self.lbl_transcript = QLabel("")
        self.lbl_transcript.setWordWrap(True)
        self.lbl_transcript.setObjectName("mono")
        gl.addWidget(self.lbl_transcript)
        pl.addWidget(g)

        g = QGroupBox("PROCEDURE")
        gl = QVBoxLayout(g)
        self.cb_procedure = QComboBox()
        for pid, name in self.core.procedures.names():
            self.cb_procedure.addItem(name, pid)
        gl.addWidget(self.cb_procedure)
        row = QHBoxLayout()
        self.btn_proc_start = QPushButton("START")
        self.btn_proc_start.setObjectName("primary")
        self.btn_proc_start.clicked.connect(lambda: self._proc_cmd("start"))
        self.btn_proc_back = QPushButton("◀ Back")
        self.btn_proc_back.clicked.connect(lambda: self._proc_cmd("back"))
        self.btn_proc_next = QPushButton("Next ▶")
        self.btn_proc_next.clicked.connect(lambda: self._proc_cmd("next"))
        self.btn_proc_stop = QPushButton("Stop")
        self.btn_proc_stop.setObjectName("danger")
        self.btn_proc_stop.clicked.connect(lambda: self._proc_cmd("stop procedure"))
        for b in (self.btn_proc_start, self.btn_proc_back, self.btn_proc_next, self.btn_proc_stop):
            row.addWidget(b)
        gl.addLayout(row)
        self.lbl_step = QLabel("No procedure running.")
        self.lbl_step.setWordWrap(True)
        self.lbl_step.setObjectName("mono")
        gl.addWidget(self.lbl_step)
        pl.addWidget(g)

        # ---- surgical case + count ----
        self.surgical = SurgicalPanel(self.core)
        pl.addWidget(self.surgical)

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
        bus.subscribe(Topic.VOICE_EVENT, lambda e: self.voice_signal.emit(e.payload["event"]))
        bus.subscribe(Topic.PROCEDURE_STEP, lambda e: self.step_signal.emit(e.payload["step"], e.payload["index"], e.payload["total"]))
        bus.subscribe(Topic.SAFETY_ALERT, lambda e: self.surgical.on_alert(e.payload["alert"]))

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

    # ---- voice ---------------------------------------------------------------------------------------
    def _toggle_listen(self, on: bool) -> None:
        if on:
            ok, msg = self.core.start_voice()
            if not ok:
                self.btn_listen.setChecked(False)
                self.lbl_voice_state.setText(f"Voice: {msg}")
                QMessageBox.warning(self, "BRIDGE voice", msg + "\n\nBRIDGE uses this computer's default microphone. "
                                    "Pick a specific one under Settings > Microphone if that is not the right input.")
                return
            self.btn_listen.setText("🎙  LISTENING")
        else:
            self.core.stop_voice()
            self.btn_listen.setText("🎙  LISTEN")
            self.lbl_voice_state.setText("Voice: off")

    def _on_voice_event(self, ev) -> None:
        state = ev.state.value
        self.lbl_voice_state.setText(f"Voice: {state}" + (f"  ({ev.detail})" if ev.detail else ""))
        if ev.transcript is not None and ev.transcript.text:
            self.lbl_transcript.setText(f"> {ev.transcript.text}" + (f"\n  {ev.response}" if ev.response else ""))
        if ev.response:
            self.lbl_ai_result.setText(ev.response)

    def _on_wake_toggle(self, on: bool) -> None:
        self.core.settings.voice.require_wake_word = on
        if self.core.voice is not None:
            self.core.voice.require_wake_word = on

    def _on_speak_toggle(self, on: bool) -> None:
        self.core.settings.voice.tts_enabled = on
        if self.core._tts is not None and not on:
            from bridge.voice.tts import SilentTTS

            self.core._tts = SilentTTS()
            if self.core.voice is not None:
                self.core.voice.tts = self.core._tts
        elif on:
            from bridge.voice.tts import build_tts

            v = self.core.settings.voice
            self.core._tts = build_tts(True, v.tts_rate, v.tts_voice)
            if self.core.voice is not None:
                self.core.voice.tts = self.core._tts

    def _ptt_pressed(self) -> None:
        """Push-to-talk: capture from the mic until release, then run one command."""
        from bridge.voice.audio import MicrophoneSource

        if self.core.voice is None or not self.core.voice.running:
            try:
                self._ptt_source = MicrophoneSource(self.core.settings.voice.mic_device)
            except RuntimeError as e:
                self.lbl_voice_state.setText(f"Voice: {e}")
                return
            self.lbl_voice_state.setText("Voice: push-to-talk (recording)")
            self._ptt_chunks = []
            import threading

            def rec():
                for c in self._ptt_source.chunks():
                    self._ptt_chunks.append(c)

            self._ptt_thread = threading.Thread(target=rec, daemon=True)
            self._ptt_thread.start()

    def _ptt_released(self) -> None:
        src = getattr(self, "_ptt_source", None)
        if src is None:
            return
        src.close()
        self._ptt_source = None
        import numpy as np
        import time

        from bridge.voice.audio import Utterance

        chunks = getattr(self, "_ptt_chunks", [])
        if not chunks:
            return
        samples = np.concatenate(chunks)
        utt = Utterance(samples, src.sample_rate, time.time(), len(samples) / src.sample_rate, 0.0)
        if self.core.voice is None:
            ok, msg = self.core.start_voice()
            if ok:
                self.core.stop_voice()  # we only needed the STT/TTS objects
        va = self.core.voice
        if va is None:
            from bridge.voice.assistant import VoiceAssistant
            from bridge.voice.stt import build_stt
            from bridge.voice.tts import build_tts

            try:
                stt = build_stt(self.core.settings.voice.stt_provider, self.core.config.secrets.gemini_api_key, self.core.settings.ai.model)
            except Exception as e:  # noqa: BLE001
                self.lbl_voice_state.setText(f"Voice: {e}")
                return
            if self.core._tts is None:
                v = self.core.settings.voice
                self.core._tts = build_tts(v.tts_enabled, v.tts_rate, v.tts_voice)
            va = VoiceAssistant(lambda: src, stt, self.core._tts, self.core.handle_spoken,
                                wake_word=self.core.settings.voice.wake_word, require_wake_word=False,
                                on_event=lambda e: self.voice_signal.emit(e))
        run_in_background(lambda: va.process_utterance(utt), lambda r: None)

    # ---- procedures ----------------------------------------------------------------------------------
    def _proc_cmd(self, word: str) -> None:
        if word == "start":
            pid = self.cb_procedure.currentData()
            proc = self.core.procedures.get(pid)
            if proc is None:
                return
            if not self.core.state.is_ready:
                self.statusBar().showMessage("Spatial calibration required.", 5000)
                return
            self.lbl_step.setText(f"Starting {proc.name}...")
            run_in_background(lambda: self.core.guide.start(proc), self._proc_reply)
        else:
            run_in_background(lambda: self.core.handle_spoken(word), self._proc_reply)

    def _proc_reply(self, reply) -> None:
        if reply:
            self.lbl_ai_result.setText(str(reply))
            self.core.speak(str(reply))

    def _on_step(self, step, index: int, total: int) -> None:
        if step is None:
            self.lbl_step.setText("Procedure complete." if total and index >= total else "No procedure running.")
        else:
            self.lbl_step.setText(f"Step {index + 1}/{total}: {step.instruction}" + (f"\n⚠ {step.caution}" if step.caution else ""))

    # ---- registration trim ---------------------------------------------------------------------------
    def keyPressEvent(self, event) -> None:  # noqa: N802
        if self.btn_validate.isChecked() and self.core.state.is_ready:
            step = 10.0 if event.modifiers() & Qt.KeyboardModifier.ShiftModifier else 1.0
            moves = {Qt.Key.Key_Left: (-step, 0), Qt.Key.Key_Right: (step, 0), Qt.Key.Key_Up: (0, -step), Qt.Key.Key_Down: (0, step)}
            if event.key() in moves:
                dx, dy = self.core.nudge_registration(*moves[event.key()])
                self.lbl_trim.setText(f"Registration trim: {dx:.0f}, {dy:.0f} px  (arrow keys nudge, Shift = ×10, R = reset)")
                return
            if event.key() == Qt.Key.Key_R:
                self.core.reset_registration_trim()
                self.lbl_trim.setText("Registration trim: 0, 0 px  (arrow keys nudge, Shift = ×10, R = reset)")
                return
        super().keyPressEvent(event)

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
        self.surgical.refresh()

    def _set_status(self, label: QLabel, text: str, ok: bool) -> None:
        label.setText(text)
        label.setObjectName("status_ok" if ok else "status_warn")
        label.style().unpolish(label)
        label.style().polish(label)

    def _update_banner(self) -> None:
        self.lbl_banner.setText(READY if self.core.state.is_ready else WELCOME)
        dx, dy = self.core.registration_trim
        self.lbl_trim.setText(f"Registration trim: {dx:.0f}, {dy:.0f} px  (test mode: arrow keys nudge, Shift = ×10, R = reset)")
        if self.core.state.is_ready and self.core.settings.voice.enabled and not self.btn_listen.isChecked() \
                and self.core.state.mode == "physical" and not getattr(self, "_voice_autostarted", False):
            self._voice_autostarted = True
            self.btn_listen.setChecked(True)

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
