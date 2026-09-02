"""BRIDGE entry point.

    bridge                  # physical mode
    bridge --simulation     # hardware-free simulation
    bridge --settings path/to/settings.yaml
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="bridge", description="BRIDGE — Universal Spatial AI Projection Platform")
    p.add_argument("--simulation", action="store_true", help="start in hardware-free simulation mode")
    p.add_argument("--settings", default="settings.yaml", help="path to settings.yaml")
    p.add_argument("--log-level", default=None, help="DEBUG, INFO, WARNING, ERROR")
    p.add_argument("--headless-selftest", action="store_true",
                   help="run a simulation calibration + query without a window, print the result, exit")
    return p.parse_args(argv)


def build_core(args: argparse.Namespace):
    from bridge.app.application import BridgeCore
    from bridge.app.config import ConfigManager
    from bridge.app.logging_setup import setup_logging

    config = ConfigManager(Path(args.settings))
    if args.simulation:
        config.settings.mode = "simulation"
    level = args.log_level or config.settings.log_level
    setup_logging(level, config.settings.log_dir)
    return BridgeCore(config)


def selftest(core) -> int:
    log = logging.getLogger("bridge.selftest")
    core.enter_simulation()
    res = core.calibrate()
    print(res.status_text())
    if not res.success:
        return 2
    r = core.ask("Where is the screwdriver?")
    print(f"ask -> ok={r.ok} strategy={r.strategy} targets={r.n_targets} message={r.message}")
    core.shutdown()
    log.info("Self-test finished")
    return 0 if r.ok else 3


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.headless_selftest:
        return selftest(build_core(args))
    from PySide6.QtWidgets import QApplication

    from bridge.ui.main_window import MainWindow
    from bridge.ui.widgets import STYLESHEET

    app = QApplication.instance() or QApplication(sys.argv)
    app.setApplicationName("BRIDGE")
    app.setStyleSheet(STYLESHEET)
    core = build_core(args)
    win = MainWindow(core)
    win.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
