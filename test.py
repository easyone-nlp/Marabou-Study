#!/usr/bin/env python3
"""Convenience wrapper for Assignment 3 Problem 2 verification."""

from pathlib import Path
import os
import runpy
import sys


if __name__ == "__main__":
    problem2_dir = Path(__file__).resolve().parent / "problem2"
    sys.path.insert(0, str(problem2_dir))
    os.chdir(problem2_dir)
    runpy.run_path(str(problem2_dir / "test.py"), run_name="__main__")
