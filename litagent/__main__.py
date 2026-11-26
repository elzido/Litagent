try:
    from .cli import main
except ImportError:              # запуск как `python litagent/` или `python litagent/__main__.py` вместо `python -m litagent`
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from litagent.cli import main

raise SystemExit(main())
