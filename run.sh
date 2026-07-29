#!/bin/sh
# Start StoryPal. PYTHONPATH=src makes the package importable without
# relying on the venv's .pth files — macOS (iCloud sync in ~/Documents)
# keeps flagging those as hidden, and Python >= 3.12.10 skips hidden
# .pth files entirely.
cd "$(dirname "$0")"
PYTHONPATH=src exec .venv/bin/uvicorn storypal.api.main:app --reload "$@"
