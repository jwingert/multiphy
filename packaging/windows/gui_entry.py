"""Windowed entry point used by the installer and desktop shortcut."""
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()
    from multiphy.desktop import gui_main
    raise SystemExit(gui_main())
