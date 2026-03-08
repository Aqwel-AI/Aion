#!/usr/bin/env python3
"""
Aqwel-Aion - Code Snippets
==========================

Extract Python comments, function names, and class names from source code via regex.
"""

import re


def extract_comments(code: str):
    """Return list of lines that are Python comments (# ...)."""
    return re.findall(r"#.*", code)


def extract_functions(code: str):
    """Return list of function names from def name(...) in code."""
    return re.findall(r"def (.+?)\(", code)


def extract_class_defs(code: str):
    """Return list of class names from class name(...) in code."""
    return re.findall(r"class (.+?)\(", code)