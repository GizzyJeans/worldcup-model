"""Filesystem anchors so the CLIs work from any working directory.

Every path is resolved relative to this package, not the current directory,
so `python E:/ClaudeCode/predict.py ...` works from any session/cwd.
"""

from __future__ import annotations

import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
MODEL_PATH = os.path.join(ROOT, "model.json")
ODDS_DIR = os.path.join(ROOT, "odds_data")
