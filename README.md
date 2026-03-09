# Aqwel-Aion

**Aqwel-Aion v0.1.8 — Complete AI Research and Development Library**

Aion is an open-source Python utility library by Aqwel AI for AI research, machine learning development, and advanced data science workflows. It provides mathematical operations, algorithm utilities, visualization, model evaluation, documentation generation, and development tooling in a single package.

---

## Team Aqwel AI

| Name | Role | GitHub | LinkedIn |
|------|------|--------|----------|
| Aksel Aghajanyan | CEO & Main Developer, AI Researcher  | [@Aksel588](https://github.com/Aksel588) | [Aksel Aghajanyan](https://www.linkedin.com/in/aksel-aghajanyan/) |
| Georgi Martirosyan | Developer | [@MrGeorge084](https://github.com/MrGeorge084) | [Georgi Martirosyan](https://www.linkedin.com/in/georgi-martirosyan-9038a43a6/) |
| Arthur Karapetyan | Developer | [@ArthurKarapetyan17](https://github.com/ArthurKarapetyan17) | [Arthur Karapetyan](https://www.linkedin.com/in/arthur-karapetyan-53b920385/) |
| David Avanesyan | Developer | [@dav1t1](https://github.com/dav1t1) | [Davit Avanesyan](https://www.linkedin.com/in/դավիթ-ավանեսյան-9a6a733a4/) |

**Author:** Aksel Aghajanyan · **Developed by:** Aqwel AI Team

---

## Table of Contents

- [Team Aqwel AI](#team-aqwel-ai)
- [Overview](#overview)
- [Architecture and structure](#architecture-and-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [Features](#features)
- [Usage Examples](#usage-examples)
- [Module Reference](#module-reference)
- [Supported Languages](#supported-languages)
- [Documentation and Resources](#documentation-and-resources)
- [What shows on GitHub](#what-shows-on-github)
- [Contributing](#contributing)
- [Author and License](#author-and-license)
- [Library Statistics](#library-statistics)

---

## Overview

Aion consolidates common research and development tasks into a consistent API: mathematics and statistics, search and array algorithms, 1D/2D and training visualization, text embeddings, prompt engineering, code analysis, model evaluation, PDF documentation generation, file management, real-time file watching, and Git integration. The library is designed for use in scripts, notebooks, and production pipelines, with optional dependencies so you can install only what you need.

---

## Architecture and structure

### High-level design

- **Single package:** All public API lives under the `aion` package. Import via `import aion` or `from aion import maths, algorithms, visualization, former`, etc.
- **Flat core + subpackages:** Core modules (maths, code, embed, evaluate, files, git, parser, pdf, prompt, snippets, text, utils, watcher, cli) are top-level modules inside `aion`. Subpackages: `aion.algorithms` (search, arrays, graphs), `aion.visualization` (arrays, matrices, training), and `aion.former` (transformer training: models, training, datasets, visualization).
- **Optional dependencies:** Heavy features (embeddings, PDF generation, visualization, advanced maths) use optional imports and extras (`[ai]`, `[docs]`, `[full]`). Core behaviour works with base install.
- **Entry point:** Package version and metadata are on `aion`; the CLI is exposed via `aion.cli` or the `main` entry point (e.g. `python -m aion.cli`).

### Directory structure

```
aion/                          # Root package
├── __init__.py                # Version, exports: maths, algorithms, visualization, embed, evaluate, code, prompt, snippets, pdf, parser, files, watcher, git, utils, text, cli
├── maths.py                   # Mathematics, statistics, linear algebra, ML helpers, signal processing
├── code.py                    # Code analysis and quality (explain, extract, complexity, docstrings, smells)
├── embed.py                   # Text embeddings and similarity (optional: sentence-transformers)
├── evaluate.py                # Classification/regression metrics, file-based evaluation
├── files.py                   # File and directory operations
├── git.py                     # Git integration (optional: gitpython)
├── parser.py                  # Language detection and code parsing
├── pdf.py                     # Documentation generation (PDF, text, Markdown; optional: reportlab)
├── prompt.py                  # Prompt templates and utilities
├── snippets.py                # Code snippet utilities
├── text.py                    # Text processing
├── utils.py                   # General utilities
├── watcher.py                 # Real-time file change monitoring
├── cli.py                     # Command-line interface
    ├── algorithms/                # Subpackage: search, arrays, graphs
    │   ├── __init__.py            # Exports: binary_search, lower_bound, upper_bound, flatten_array, chunk_array, ...
    │   ├── search.py             # Binary/jump/exponential/linear search, bounds, first/last occurrence, peaks
    │   ├── arrays.py              # flatten, chunk, sliding_window, rolling_sum, remove_duplicates, pad, ...
    │   ├── graphs.py              # Placeholder (future: BFS, DFS, Dijkstra, toposort)
    │   ├── README.md              # Algorithm package documentation
    │   └── examples/              # Jupyter notebooks: search, array utilities
    │       ├── 01_search_algorithms.ipynb
    │       ├── 02_array_utilities.ipynb
    │       └── README.md
    ├── former/                    # Subpackage: transformer training (Aion Former)
    │   ├── __init__.py            # Exports: Transformer, Trainer, TextDataset, plot_attention_map, ...
    │   ├── core/                  # Tensor, autograd, matmul, softmax, layer_norm, attention ops
    │   ├── models/                # Embedding, positional encoding, multi-head attention, feed-forward, transformer
    │   ├── training/               # Trainer, Adam, cross_entropy_loss
    │   ├── datasets/              # SimpleTokenizer, TextDataset, create_dataloader
    │   ├── visualization/          # plot_attention_map, plot_training_metrics, plot_weight_spectrum
    │   ├── experiments/           # train_small_model.py, config.yaml
    │   ├── examples/              # attention_demo, text_generation
    │   └── docs/                  # architecture.md
    └── visualization/             # Subpackage: 1D/2D and training plots
    ├── __init__.py            # Exports: plot_array, plot_histogram, plot_confusion_matrix, plot_training_history, ...
    ├── arrays.py              # 1D plots (array, histogram, scatter, running mean, boxplot, density, cdf, ...)
    ├── matrices.py            # 2D plots (heatmap, confusion, surface, contour, correlation, attention, sparsity)
    ├── training.py            # Training plots (history, metric, train vs val, learning rate, early stopping, ...)
    ├── utils.py               # finalize_plot, save_plot
    ├── README.md              # Visualization documentation
    ├── examples/              # Jupyter notebooks: array, matrix, training
    │   ├── 01_array_visualization.ipynb
    │   ├── 02_matrix_visualization.ipynb
    │   ├── 03_training_visualization.ipynb
    │   └── README.md
    └── examples_visualization/  # Example output images (PNG)
```

Repository root also contains `example.py` (runnable visualization/algorithms demo), `main.py` (CLI entry), `pyproject.toml` / `setup.py`, `requirements.txt`, `CONTRIBUTING.md`, `CHANGELOG.md`, and `LICENSE`.

### Design principles

- **Explicit imports:** Subpackages re-export their main symbols in `__init__.py`; you can use `from aion.algorithms import binary_search` or `from aion.algorithms.search import binary_search`.
- **Backend-safe visualization:** Plot functions return matplotlib Figures and accept `show=False`; in headless environments they do not require a display.
- **Layered dependencies:** Core and algorithms depend mainly on numpy and the standard library; embed, pdf, and visualization add optional dependencies when used.

---

## Requirements

- **Python:** 3.8 or higher (3.9, 3.10, 3.11, 3.12 supported).
- **pip:** For installing the package and optional extras.
- **Core runtime:** `numpy>=1.21.0`, `watchdog>=2.1.0`, `gitpython>=3.1.0` (optional for Git features).
- **Optional:** SciPy, scikit-learn, pandas, matplotlib, ReportLab, sentence-transformers, PyTorch, etc. See [Installation](#installation) for extras.

A virtual environment (e.g. `venv` or `conda`) is recommended to isolate dependencies.

---

## Installation

### Base install (required dependencies only)

```bash
pip install aqwel-aion
```

This installs the core package with numpy, watchdog, and gitpython. Enough for maths, algorithms, parser, files, utils, text, and most of the code and evaluate modules.

### Optional dependency groups

```bash
pip install aqwel-aion[viz]   # Visualization (matplotlib, seaborn)
pip install aqwel-aion[former] # Transformer training (Aion Former: matplotlib, pyyaml)
pip install aqwel-aion[ai]     # ML stack: scipy, scikit-learn, pandas, matplotlib, transformers, torch, sentence-transformers, openai
pip install aqwel-aion[docs]   # PDF/docs: reportlab, pillow
pip install aqwel-aion[full]   # All optional dependencies including seaborn, faiss-cpu
pip install aqwel-aion[dev]    # Development: pytest, black, flake8
```

### Editable install (for development)

```bash
git clone https://github.com/aqwelai/aion.git
cd aion
pip install -e .[dev,full]
```

### Step-by-step (first-time setup)

1. Create and activate a virtual environment (recommended):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

2. Upgrade pip and install the package:

   ```bash
   pip install --upgrade pip
   pip install aqwel-aion
   ```

3. For visualization and full ML/docs, use extras:

   ```bash
   pip install aqwel-aion[full]
   ```

4. Verify the install:

   ```bash
   python -c "import aion; print(aion.__version__)"
   ```

---

## Getting Started

### Verify installation

```python
import aion
print(aion.__version__)  # 0.1.8
print(aion.__author__)      # Aksel Aghajanyan
print(aion.__developer__)   # Aqwel AI Team
```

### Minimal example (no optional deps)

```python
import aion

# Mathematics (uses numpy; no optional deps)
r = aion.maths.addition(2, 3)           # 5
r = aion.maths.mean([1.0, 2.0, 3.0])    # 2.0
r = aion.maths.determinant([[1, 2], [3, 4]])  # -2.0

# Algorithms (stdlib only from aion.algorithms)
idx = aion.algorithms.binary_search([1, 3, 5, 7, 9], 7)  # 4
flat = aion.algorithms.flatten_array([[1, 2], [3, 4]])   # [1, 2, 3, 4]
```

### Run the CLI (if installed)

```bash
python -m aion.cli
# or, if entry point is installed:
aion
```

The repository includes an `example.py` in the project root that demonstrates visualization and algorithms; run it with `python example.py` after installing with matplotlib available.

---

## Features

### Mathematics and Statistics

- **71+ mathematical functions** for linear algebra, statistics, and numerical computation.
- **Linear algebra:** vectors, matrices, eigenvalues, SVD, determinant, inverse; optional SciPy for matrix exponential and logarithm with NumPy fallbacks.
- **Statistics:** correlation, regression, probability distributions, hypothesis testing, descriptive statistics.
- **Machine learning helpers:** activation functions (sigmoid, ReLU, tanh, etc.), loss functions, distance metrics.
- **Signal processing:** FFT, convolution, filtering, frequency analysis.
- **Trigonometry, logarithms, and basic arithmetic** with support for scalars, lists, and string numerals.

### Algorithms

- **Search (aion.algorithms.search):** Binary search, lower_bound, upper_bound; jump search, exponential search, linear search; first/last occurrence; is_sorted, find_peak_element; rotated sorted array search, ternary search, interpolation search.
- **Arrays (aion.algorithms.arrays):** flatten_array, flatten_deep, chunk_array, pairwise, sliding_window, rolling_sum, remove_duplicates, moving_avarage, pad_array.
- **Graphs:** Placeholder for future graph traversal and shortest-path algorithms (BFS, DFS, Dijkstra, toposort).
- Jupyter example notebooks in `aion/algorithms/examples/` with full API coverage and explanations.

### Visualization

- **1D arrays:** plot_array, plot_histogram, plot_scatter, plot_multiple_arrays, plot_array_with_mean, plot_running_mean; plot_boxplot, plot_density, plot_cdf; plot_error_bars, plot_rolling_std, plot_min_max_band; plot_autocorrelation, plot_quantiles, plot_scatter_with_fit, plot_dual_axis.
- **2D matrices:** plot_matrix_heatmap, plot_confusion_matrix (raw and normalized), plot_matrix_surface, plot_matrix_contour, plot_matrix_with_values; plot_correlation_matrix, plot_similarity_matrix; plot_matrix_histogram, plot_masked_heatmap; plot_attention_map, plot_matrix_sparsity.
- **Training:** plot_training_history, plot_metric, plot_train_vs_val, plot_learning_rate, plot_metric_with_best, plot_metrics_grid, plot_confidence_band, plot_early_stopping, plot_epoch_time.
- All plotting functions return a matplotlib Figure; use `aion.visualization.utils.save_plot(fig, path)` to save. Example notebooks in `aion/visualization/examples/`.

### AI Research and ML

- **Text embeddings:** Sentence-transformers integration and vector operations (e.g. cosine similarity).
- **Prompt engineering:** Specialized AI prompt templates and utilities for research workflows.
- **Code analysis:** Structural explanation, function/class/import extraction, comment stripping, cyclomatic complexity, docstring extraction, operator counts, code smell detection.
- **Model evaluation:** Classification metrics (accuracy, precision, recall, F1, confusion matrix, ROC-AUC), regression metrics (MSE, RMSE, MAE, R²); file-based evaluation (JSON/CSV) with automatic task detection.

### Documentation Generation

- **PDF and text:** API reference, user guides, changelogs, module dependency reports; configurable branding (colors, fonts, logo).
- **Markdown:** API documentation with table of contents.
- **Exports:** Machine-readable API index (JSON/CSV), function lists. ReportLab is optional; entry points fall back to text or Markdown when ReportLab is not installed.

### Development and Infrastructure

- **File management:** Create, move, copy, delete; directory listing and organization helpers.
- **Code parser:** Language detection and detailed analysis for 30+ programming languages (see [Supported Languages](#supported-languages)).
- **Real-time monitoring:** File change detection and callbacks via the watcher module.
- **Git integration:** Status, commit history, branches, diffs, file history (optional dependency: GitPython).
- **Utilities and CLI:** General helpers and command-line interface for common operations.

### Aion Former — Transformer training

- **Decoder-only (GPT-style) transformers** with NumPy-backed autograd: no PyTorch/TF required for small-scale experiments.
- **Core:** `Tensor` with gradient tracking; `matmul`, `softmax`, `layer_norm`, `relu`, scaled dot-product attention.
- **Model:** Embedding, sinusoidal positional encoding, multi-head attention, feed-forward blocks, pre-norm stack, LM head.
- **Training:** Cross-entropy loss, Adam optimizer, `Trainer` with `train_step` / `train_epoch`.
- **Data:** Character- or word-level tokenizer, sliding-window text dataset, batch loader.
- **Visualization:** Attention heatmaps (per head/layer), training loss over epochs, weight eigenvalue/singular-value spectrum.
- **Install:** `pip install aqwel-aion[former]`. Run: `python -m aion.former.experiments.train_small_model`, `python -m aion.former.examples.attention_demo`, `python -m aion.former.examples.text_generation`.

---

## Usage Examples

The following examples are drawn from the library and the project’s `example.py` and notebooks. They show how to use the main modules after installation.

### Mathematics and statistics

```python
import aion

# Basic arithmetic and statistics
aion.maths.addition(10, 5)
aion.maths.mean([1, 2, 3, 4, 5])
aion.maths.variance([1, 2, 3, 4, 5])
aion.maths.std_dev([1, 2, 3, 4, 5])
aion.maths.correlation([1, 2, 3, 4], [2, 4, 6, 8])
aion.maths.min_max_scale([1, 2, 3, 4, 5])
aion.maths.z_score([1.0, 2.0, 3.0, 4.0, 5.0])

# Linear algebra
aion.maths.determinant([[1, 2], [3, 4]])
aion.maths.dot_product([1, 2, 3], [4, 5, 6])
aion.maths.transpose([[1, 2], [3, 4], [5, 6]])
aion.maths.matrix_multiply([[1, 2], [3, 4]], [[5, 6], [7, 8]])
aion.maths.normalize_vector([3, 4], norm="l2")

# Activations and ML helpers
aion.maths.sigmoid([0, 1, -1])
aion.maths.relu([-1, 0, 1, 2])
aion.maths.softmax([1.0, 2.0, 3.0])
```

### Algorithms: search and arrays

```python
import aion
from aion.algorithms import binary_search, lower_bound, upper_bound, flatten_array, chunk_array
from aion.algorithms.search import is_sorted, jump_search, find_peak_element, exponential_search
from aion.algorithms.arrays import sliding_window, rolling_sum, remove_duplicates

# Search (sorted list required for binary_search, lower_bound, upper_bound)
arr = [10, 20, 30, 40, 50, 60, 70]
binary_search(arr, 50)    # 4
lower_bound(arr, 35)     # 2
upper_bound(arr, 50)     # 5
is_sorted([1, 2, 3, 4])  # True
jump_search([1, 3, 5, 7, 9], step=2, target=7)
exponential_search([1, 3, 5, 7, 9], 9)
find_peak_element([1, 3, 2, 4, 1])  # [3, 4]

# Array utilities
flatten_array([[1, 2], [3, 4], [5]])
chunk_array([1, 2, 3, 4, 5, 6, 7], size=3)
list(sliding_window([1, 2, 3, 4, 5, 6], 3))
rolling_sum([1, 2, 3, 4, 5, 6], 3)
remove_duplicates([3, 1, 2, 1, 4, 2, 3])
```

### Visualization (requires matplotlib)

```python
import aion
from aion.visualization import (
    plot_array,
    plot_histogram,
    plot_scatter,
    plot_multiple_arrays,
    plot_array_with_mean,
    plot_running_mean,
    plot_matrix_heatmap,
    plot_confusion_matrix,
    plot_training_history,
)
from aion.visualization.utils import save_plot

# 1D plots (use show=False in scripts to avoid blocking)
fig = plot_array([1, 3, 2, 5, 4], title="Basic Array Plot", show=False)
save_plot(fig, "example_array.png")

fig = plot_histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], bins=4, title="Value Distribution", show=False)
save_plot(fig, "example_histogram.png")

fig = plot_scatter(x=[1, 2, 3, 4, 5], y=[5, 4, 3, 2, 1], title="Scatter", show=False)
save_plot(fig, "example_scatter.png")

fig = plot_multiple_arrays(
    arrays=[[1, 2, 3, 4], [4, 3, 2, 1]],
    labels=["Increasing", "Decreasing"],
    title="Multiple Arrays",
    show=False,
)
save_plot(fig, "example_multiple_arrays.png")

fig = plot_array_with_mean([10, 12, 9, 11, 10, 13], title="Array with Mean", show=False)
save_plot(fig, "example_array_mean.png")

fig = plot_running_mean(
    [15, 16, 14, 17, 18, 20, 19, 21, 22, 20, 18, 17],
    window_size=6,
    show=False,
)
save_plot(fig, "example_running_mean.png")

# Matrix and training
fig = plot_matrix_heatmap([[1, 2, 3], [4, 5, 6], [7, 8, 9]], title="Matrix Heatmap", show=False)
save_plot(fig, "example_matrix_heatmap.png")

fig = plot_confusion_matrix(
    [[50, 5], [8, 37]],
    labels=["Negative", "Positive"],
    title="Confusion Matrix",
    show=False,
)
save_plot(fig, "example_confusion_matrix.png")

history = {"loss": [1.0, 0.7, 0.4, 0.25], "val_loss": [1.1, 0.8, 0.5, 0.3], "accuracy": [0.5, 0.65, 0.78, 0.85]}
fig = plot_training_history(history, show=False)
save_plot(fig, "example_training_history.png")
```

### Model evaluation

```python
import aion

# In-memory metrics
y_true = [0, 1, 1, 0, 1]
y_pred = [0, 1, 0, 0, 1]
metrics = aion.evaluate.calculate_classification_metrics(y_pred, y_true)
# accuracy, precision, recall, f1_score, etc.

pred_vals = [1.2, 2.1, 3.0]
true_vals = [1.0, 2.0, 3.2]
reg_metrics = aion.evaluate.calculate_regression_metrics(pred_vals, true_vals)
# mse, rmse, mae, r2

# File-based evaluation (JSON or CSV)
file_metrics = aion.evaluate.evaluate_predictions("preds.json", "answers.json")
```

### Code analysis

```python
import aion

source = """
def train_model(x, y):
    return x + y

class Trainer:
    pass
"""
aion.code.explain_code(source)
aion.code.extract_functions(source)
aion.code.extract_classes(source)
aion.code.extract_imports(source)
aion.code.strip_comments(source)
aion.code.analyze_complexity(source)
aion.code.extract_docstrings(source)
aion.code.count_operators(source)
aion.code.find_code_smells(source)
```

### File management and watcher

```python
import aion

aion.files.create_empty_file("research.txt")
# Other helpers: move, copy, delete, list files, etc.

def on_change(path):
    print("Changed:", path)
aion.watcher.watch_file_for_changes("data.csv", on_change_callback=on_change)
```

### Documentation generation (optional: reportlab for PDF)

```python
import aion

aion.pdf.generate_complete_documentation("my_docs")
aion.pdf.create_api_documentation("api_ref.pdf")
aion.pdf.create_user_guide_pdf("user_guide.pdf")
aion.pdf.create_changelog_pdf("changelog.pdf")
# Many more: create_api_documentation_md, create_module_dependency_doc, export_api_index, etc.
```

### Embeddings (optional: sentence-transformers)

```python
import aion

vec = aion.embed.embed_text("Machine learning research")
sim = aion.embed.cosine_similarity(vec1, vec2)
```

### Git (optional: gitpython)

```python
import aion

manager = aion.git.GitManager(".")
status = manager.status()
commits = manager.get_commit_history(limit=10)
```

### Aion Former — transformer training (optional: pip install aqwel-aion[former])

```python
import aion
from aion.former import Transformer, Trainer
from aion.former.datasets import create_dataloader
from aion.former.visualization import plot_attention_map, plot_training_metrics

text = "Your training corpus here. " * 100
dataset, get_batch = create_dataloader(text, seq_length=64, batch_size=32, level="char")
model = Transformer(
    vocab_size=dataset.vocab_size,
    embed_dim=128,
    num_heads=4,
    num_layers=2,
    max_seq_len=64,
)
trainer = Trainer(model, lr=0.001)
for epoch in range(10):
    loss = trainer.train_epoch(get_batch, 50)
    print(f"Epoch {epoch + 1}  loss = {loss:.4f}")
plot_training_metrics(trainer.history)
```

Run from command line: `python -m aion.former.experiments.train_small_model`, `python -m aion.former.examples.attention_demo`, `python -m aion.former.examples.text_generation`.

---

## Module Reference

| Module | Description |
|--------|-------------|
| `aion.maths` | Mathematics, statistics, linear algebra, ML helpers, signal processing. |
| `aion.algorithms` | Search (binary, bounds, jump, exponential, etc.) and array utilities (flatten, chunk, window, dedupe, rolling sum, pad). |
| `aion.visualization` | 1D/2D and training plots; heatmaps, confusion matrices, attention maps; save_plot utility. |
| `aion.former` | Transformer training: Transformer, Trainer, TextDataset, tokenizer, attention/training/weight-spectrum plots. Install with `[former]`. |
| `aion.embed` | Text embeddings and vector similarity (optional: sentence-transformers). |
| `aion.evaluate` | Classification and regression metrics; file-based evaluation. |
| `aion.code` | Code explanation, extraction, complexity, docstrings, code smells. |
| `aion.prompt` | Prompt templates and utilities. |
| `aion.snippets` | Code snippet utilities. |
| `aion.pdf` | API/user-guide/changelog generation (PDF, text, Markdown); optional ReportLab. |
| `aion.parser` | Language detection and code parsing (30+ languages). |
| `aion.files` | File and directory operations. |
| `aion.watcher` | Real-time file change monitoring. |
| `aion.git` | Git repository operations (optional: GitPython). |
| `aion.utils` | General utilities. |
| `aion.text` | Text processing. |
| `aion.cli` | Command-line interface. |

Package entry point and version:

```python
import aion
print(aion.__version__)  # 0.1.8
```

---

## Supported Languages

The parser and code analysis modules support the following (among others):

**Programming languages:** Python, JavaScript, TypeScript, Java, C, C++, C#, Go, Rust, Swift, Kotlin, Scala, Haskell, PHP, Ruby, Perl, Lua, Julia, R, MATLAB, Clojure, PowerShell, Bash.

**Markup and data:** HTML, CSS, SQL, JSON, XML, YAML, Markdown, Dockerfile, Terraform, Ansible.

See `aion.parser` and `aion.code` for language-specific behavior and APIs.

---

## Documentation and Resources

- **Official site:** [https://aqwelai.xyz/](https://aqwelai.xyz/)
- **PyPI:** [https://pypi.org/project/aqwel-aion/](https://pypi.org/project/aqwel-aion/)
- **Package metadata and URLs:** See [pyproject.toml](pyproject.toml) for project links and optional dependencies.
- **In-package docs:** Use `aion.pdf.generate_complete_documentation(output_dir)` to generate API and user-guide artifacts. Algorithm and visualization API details are in `aion/algorithms/README.md` and `aion/visualization/README.md`.
- **Example notebooks:**
  - Algorithms: `aion/algorithms/examples/` (search, array utilities).
  - Visualization: `aion/visualization/examples/` (array, matrix, training plots).
- **Changelog:** [CHANGELOG.md](CHANGELOG.md) for version history.
- **Contributing:** [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, code style, and pull request process.

---

## What shows on GitHub

This repository is open source. The following **should show** (and are committed):

| Category | What shows |
|----------|------------|
| **Docs** | `README.md`, `LICENSE`, `CHANGELOG.md`, `CONTRIBUTING.md` |
| **Config** | `pyproject.toml`, `setup.py`, `MANIFEST.in`, `requirements.txt` |
| **Source** | `aion/**/*.py`, `src/aion_core.cpp` |
| **Examples** | `example.py`, `main.py`, Jupyter notebooks in `aion/algorithms/examples/`, `aion/visualization/examples/` |
| **Example assets** | Small example images in `aion/visualization/examples_visualization/*.png` (for docs) |
| **Repo meta** | `.gitignore` |

The following **do not show** (ignored via `.gitignore`):

- Build artifacts: `build/`, `dist/`, `*.egg`, `*.egg-info/`
- Python cache: `__pycache__/`, `*.pyc`, `*.pyo`
- Virtual environments: `.venv/`, `venv/`, `env/`
- Secrets: `.env`, `.env.*` (never commit; use `.env.example` as a template if needed)
- IDE/editor: `.idea/`, `.vscode/`, `.cursor/`
- OS files: `.DS_Store`
- Test/coverage: `.coverage`, `htmlcov/`, `.pytest_cache/`, `.mypy_cache/`, `.ipynb_checkpoints/`
- Generated output: optionally `example_output/` (uncomment in `.gitignore` if you regenerate those PNGs and don’t want them on GitHub)

If something that should be hidden still appears, it was committed before being added to `.gitignore`. Remove it from tracking with `git rm -r --cached <path>` and commit.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for:

- How to report bugs and suggest features
- Development setup (`pip install -e .[dev,full]`)
- Code style (PEP 8, type hints, docstrings)
- Testing and documentation expectations
- Pull request and review process

---

## Author and License

- **Author:** Aksel Aghajanyan  
- **Developed by:** Aqwel AI Team  
- **Company Gmail:** aqwelai.company@gmail.com  
- **Copyright:** 2025 Aqwel AI  
- **License:** Apache-2.0 (see [LICENSE](LICENSE))

---

## Library Statistics

- **15+ core modules** (maths, algorithms, visualization, former, embed, evaluate, code, prompt, snippets, pdf, parser, files, watcher, git, utils, text, cli).
- **71+ mathematical functions** in the maths module.
- **Aion Former:** Decoder-only transformer training with NumPy autograd, multi-head attention, and visualization (optional `[former]` extra).
- **Full research pipeline** from data and algorithms through visualization and documentation.
- **Optional dependencies** for embeddings, PDF generation, and full ML stack; core and algorithms work with minimal dependencies (e.g. numpy, standard library).

---

Aion is designed so researchers and developers can reuse common operations without reimplementing them, from numerical and algorithmic foundations through to publication-ready documentation and plots.
