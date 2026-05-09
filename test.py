#!/usr/bin/env python3
"""Convenience wrapper for Assignment 3 Problem 2 verification."""

from pathlib import Path
import os
import sys


if __name__ == "__main__":
    problem2_dir = Path(__file__).resolve().parent / "problem2"
    sys.path.insert(0, str(problem2_dir))
    os.chdir(problem2_dir)

    from verify_marabou import main

    main()
