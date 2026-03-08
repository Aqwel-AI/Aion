#!/usr/bin/env python3
"""
Aqwel-Aion - Code Analysis and Quality Assessment
=================================================

Static analysis and introspection for source code: explain code patterns,
extract functions, classes, imports, and docstrings, strip comments,
compute cyclomatic complexity and operator counts, and detect code smells.
Uses AST-based parsing where possible and regex for broader language support.

Author: Aksel Aghajanyan
Developed by: Aqwel AI Team
License: Apache-2.0
Copyright: 2025 Aqwel AI
"""

import re
import ast
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter


def explain_code(code: str) -> str:
    """
    Produce a short, human-readable summary of code structure and patterns.

    Uses regex to detect functions, classes, imports, loops (for/while),
    conditionals, try-except blocks, and with statements, then returns a
    single sentence or a few sentences joined by ". " describing what was
    found, plus non-empty line count. Optimized for Python-like syntax but
    applicable to other languages.

    Args:
        code: Source code string (snippet or full file).

    Returns:
        Explanation string, or "Simple code without complex patterns detected."
        when nothing is matched.
    """
    explanations = []
    if re.search(r'\bfor\b.*\bin\b', code):
        explanations.append("Contains for-in loops for iteration")
    
    if re.search(r'\bwhile\b', code):
        explanations.append("Contains while loops")
        
    if re.search(r'\bdef\s+\w+\s*\(', code):
        functions = extract_functions(code)
        explanations.append(f"Defines {len(functions)} function(s): {', '.join(functions)}")
    
    if re.search(r'\bclass\s+\w+', code):
        classes = extract_classes(code)
        explanations.append(f"Defines {len(classes)} class(es): {', '.join(classes)}")
    
    if re.search(r'\bimport\b|\bfrom\b.*\bimport\b', code):
        imports = extract_imports(code)
        explanations.append(f"Imports {len(imports)} module(s)/library(s)")
    
    if re.search(r'\bif\b.*:', code):
        explanations.append("Contains conditional statements")
    
    if re.search(r'\btry\b.*\bexcept\b', code):
        explanations.append("Contains error handling (try-except)")
    
    if re.search(r'\bwith\b.*:', code):
        explanations.append("Uses context managers (with statements)")

    lines = [line.strip() for line in code.splitlines() if line.strip()]
    explanations.append(f"Contains {len(lines)} non-empty lines")
    
    if not explanations:
        return "Simple code without complex patterns detected."
    
    return ". ".join(explanations) + "."


def extract_functions(code: str) -> List[str]:
    """
    Return the names of all functions defined in the code (def name(...)).
    Uses regex; works best with Python-style source.
    """
    return re.findall(r'def\s+(\w+)\s*\(', code)


def extract_classes(code: str) -> List[str]:
    """
    Return the names of all classes defined in the code (class name).
    Uses regex; works best with Python-style source.
    """
    return re.findall(r'class\s+(\w+)', code)


def extract_imports(code: str) -> List[str]:
    """
    Return the list of module names used in import and from ... import
    statements. Duplicates are removed.
    """
    imports = []
    imports.extend(re.findall(r'^\s*import\s+([^\s,]+)', code, re.MULTILINE))
    from_imports = re.findall(r'^\s*from\s+([^\s]+)\s+import', code, re.MULTILINE)
    imports.extend(from_imports)
    return list(set(imports))


def strip_comments(code: str) -> str:
    """
    Return the code with line-ending and whole-line # comments removed.
    Does not handle # inside string literals; strips from the first # to
    end of line. Empty lines are dropped.
    """
    lines = []
    for line in code.splitlines():
        stripped = line
        if "#" in line:
            comment_pos = line.find("#")
            stripped = line[:comment_pos].rstrip()
        if stripped:
            lines.append(stripped)
    return "\n".join(lines)


def analyze_complexity(code: str) -> Dict[str, Any]:
    """
    Return a dict of simple complexity metrics: total_lines, code_lines,
    counts of functions, classes, imports, if/loops/try, and a simplified
    cyclomatic complexity (1 + if + for + while + try).
    """
    lines = code.splitlines()
    non_empty_lines = [line for line in lines if line.strip()]
    functions = len(extract_functions(code))
    classes = len(extract_classes(code))
    imports = len(extract_imports(code))
    if_count = len(re.findall(r'\bif\b', code))
    for_count = len(re.findall(r'\bfor\b', code))
    while_count = len(re.findall(r'\bwhile\b', code))
    try_count = len(re.findall(r'\btry\b', code))
    complexity = 1 + if_count + for_count + while_count + try_count
    
    return {
        'total_lines': len(lines),
        'code_lines': len(non_empty_lines),
        'functions': functions,
        'classes': classes,
        'imports': imports,
        'if_statements': if_count,
        'loops': for_count + while_count,
        'try_blocks': try_count,
        'cyclomatic_complexity': complexity
    }


def extract_docstrings(code: str) -> List[str]:
    """
    Return all non-empty triple-quoted strings (\"\"\" or \'\'\') from the
    code. May include strings that are not formal docstrings.
    """
    docstrings = []
    docstring_pattern = r'"""(.*?)"""|\'\'\'(.*?)\'\'\''
    matches = re.findall(docstring_pattern, code, re.DOTALL)
    
    for match in matches:
        docstring = match[0] if match[0] else match[1]
        if docstring.strip():
            docstrings.append(docstring.strip())
    
    return docstrings


def count_operators(code: str) -> Dict[str, int]:
    """
    Return a dict of operator counts by category: arithmetic, comparison,
    logical, assignment, bitwise. Counts are from simple token/character
    search and may include operators inside strings.
    """
    operators = {
        'arithmetic': ['+', '-', '*', '/', '//', '%', '**'],
        'comparison': ['==', '!=', '<', '>', '<=', '>='],
        'logical': ['and', 'or', 'not'],
        'assignment': ['=', '+=', '-=', '*=', '/='],
        'bitwise': ['&', '|', '^', '~', '<<', '>>']
    }
    
    counts = {}
    for category, ops in operators.items():
        count = 0
        for op in ops:
            if op.isalpha():
                count += len(re.findall(r'\b' + op + r'\b', code))
            else:
                count += code.count(op)
        counts[category] = count
    
    return counts


def find_code_smells(code: str) -> List[str]:
    """
    Return a list of short descriptions of potential code smells: long
    functions (>50 lines), deep nesting (>4 indentation levels), many
    magic numbers, long lines (>100 chars), and TODO/FIXME/HACK comments.
    Duplicates are removed.
    """
    smells = []
    functions = re.findall(r'def\s+\w+.*?(?=def|\Z)', code, re.DOTALL)
    for func in functions:
        if len(func.splitlines()) > 50:
            smells.append("Long function detected (>50 lines)")
    lines = code.splitlines()
    for line in lines:
        if len(line) - len(line.lstrip()) > 16:
            smells.append("Deep nesting detected (>4 levels)")
            break
    magic_numbers = re.findall(r'\b\d{2,}\b', code)
    if len(magic_numbers) > 5:
        smells.append("Many magic numbers detected")
    long_lines = [line for line in lines if len(line) > 100]
    if long_lines:
        smells.append(f"{len(long_lines)} long lines detected (>100 chars)")
    todos = re.findall(r'#.*TODO|#.*FIXME|#.*HACK', code, re.IGNORECASE)
    if todos:
        smells.append(f"{len(todos)} TODO/FIXME/HACK comments found")
    return list(set(smells))