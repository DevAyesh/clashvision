"""
ClashVision main application entry point.
"""

import sys
import argparse
from clashvision import __version__


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="clashvision",
        description="ClashVision: Clash of Clans intelligent tactical & farming analyzer",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without displaying graphical interface (CLI/service mode)",
    )
    parser.add_argument(
        "--mock-capture",
        action="store_true",
        help="Use synthetic offline frame capture instead of live window capture",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=15,
        help="Target capture frame rate (default: 15 FPS)",
    )
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.headless:
        print(f"ClashVision v{__version__} starting in headless mode (FPS={args.fps}, mock={args.mock_capture}).")
        return 0

    from clashvision.ui.qt_compat import QApplication
    from clashvision.ui.main_window import MainWindow
    from clashvision.core.config import AppSettings

    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv if argv is None else [sys.argv[0]] + list(argv))

    settings = AppSettings()
    if args.fps:
        settings.capture.target_fps = args.fps

    window = MainWindow(settings=settings, start_workers=True)
    window.show()

    if hasattr(app, "exec"):
        return app.exec()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
