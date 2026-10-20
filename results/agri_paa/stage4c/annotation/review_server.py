#!/usr/bin/env python
"""Standalone launcher for the annotation review UI."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
from agri.plant_gs.paa.stage4c.annotation import main

if __name__ == "__main__":
    sys.argv = ["annotation.py", "serve", "--dir", str(Path(__file__).parent)] + sys.argv[1:]
    main()
