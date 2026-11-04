"""Console entry point for the frozen application and its workers."""
import multiprocessing


if __name__ == '__main__':
    multiprocessing.freeze_support()
    from multiphy.desktop import cli_main
    raise SystemExit(cli_main())
