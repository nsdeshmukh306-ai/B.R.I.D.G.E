"""Settings dialog (writes settings.yaml)."""
from __future__ import annotations

from PySide6.QtWidgets import (
    QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLineEdit, QSpinBox, QVBoxLayout, QWidget,
)

from bridge.app.config import ConfigManager


class SettingsDialog(QDialog):
    def __init__(self, config: ConfigManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("BRIDGE — Settings")
        self.config = config
        s = config.settings
        form = QFormLayout()

        self.bg = QComboBox()
        self.bg.addItems(["white", "black", "transparent", "custom"])
        self.bg.setCurrentText(s.render.background)
        form.addRow("Projection background", self.bg)

        self.target_style = QComboBox()
        self.target_style.addItems(["pulse", "static"])
        self.target_style.setCurrentText(s.render.target_style)
        form.addRow("Target style", self.target_style)

        self.line_width = QSpinBox()
        self.line_width.setRange(1, 20)
        self.line_width.setValue(s.render.line_width)
        form.addRow("Line width (px)", self.line_width)

        self.cal_method = QComboBox()
        self.cal_method.addItems(["planar_4point", "planar_9point"])
        self.cal_method.setCurrentText(s.calibration.method)
        form.addRow("Calibration method", self.cal_method)

        self.cal_threshold = QDoubleSpinBox()
        self.cal_threshold.setRange(0.5, 100.0)
        self.cal_threshold.setValue(s.calibration.validation_threshold_px)
        form.addRow("Validation threshold (px)", self.cal_threshold)

        self.marker_radius = QSpinBox()
        self.marker_radius.setRange(6, 120)
        self.marker_radius.setValue(s.calibration.marker_radius_px)
        form.addRow("Marker radius (px)", self.marker_radius)

        self.tracker = QComboBox()
        self.tracker.addItems(["csrt", "kcf", "mosse", "mil", "color"])
        self.tracker.setCurrentText(s.tracking.backend)
        form.addRow("Tracking backend", self.tracker)

        self.ai_model = QLineEdit(s.ai.model)
        form.addRow("Gemini model", self.ai_model)

        self.ai_conf = QDoubleSpinBox()
        self.ai_conf.setRange(0.0, 1.0)
        self.ai_conf.setSingleStep(0.05)
        self.ai_conf.setValue(s.ai.min_confidence)
        form.addRow("Min AI confidence", self.ai_conf)

        self.cam_w = QSpinBox()
        self.cam_w.setRange(160, 7680)
        self.cam_w.setValue(s.camera.width)
        self.cam_h = QSpinBox()
        self.cam_h.setRange(120, 4320)
        self.cam_h.setValue(s.camera.height)
        form.addRow("Camera width", self.cam_w)
        form.addRow("Camera height", self.cam_h)

        self.log_level = QComboBox()
        self.log_level.addItems(["DEBUG", "INFO", "WARNING", "ERROR"])
        self.log_level.setCurrentText(s.log_level.upper())
        form.addRow("Log level", self.log_level)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        lay = QVBoxLayout(self)
        lay.addLayout(form)
        lay.addWidget(buttons)

    def _save(self) -> None:
        s = self.config.settings
        s.render.background = self.bg.currentText()  # type: ignore[assignment]
        s.render.target_style = self.target_style.currentText()  # type: ignore[assignment]
        s.render.line_width = self.line_width.value()
        s.calibration.method = self.cal_method.currentText()
        s.calibration.validation_threshold_px = self.cal_threshold.value()
        s.calibration.marker_radius_px = self.marker_radius.value()
        s.tracking.backend = self.tracker.currentText()  # type: ignore[assignment]
        s.ai.model = self.ai_model.text().strip() or s.ai.model
        s.ai.min_confidence = self.ai_conf.value()
        s.camera.width, s.camera.height = self.cam_w.value(), self.cam_h.value()
        s.log_level = self.log_level.currentText()
        self.config.save()
        self.accept()
