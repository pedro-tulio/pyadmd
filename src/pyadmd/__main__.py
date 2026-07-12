"""Enables ``python -m pyadmd`` to behave identically to the ``pyadmd`` console script."""

from pyadmd.cli.main import main

if __name__ == "__main__":
    main()
