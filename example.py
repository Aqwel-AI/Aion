"""
Full test of aion.algorithms, aion.visualization, and new features in v0.1.8.

This script exercises every exported function from both packages so you can
verify behaviour and see typical usage. Output is printed for algorithms;
all plots are written to example_output/ (no interactive display), so the
script is safe to run in a terminal or CI.

Contents:
  - Section 1: aion.algorithms
    - Search: binary search, lower/upper bound, is_sorted, jump/exponential/linear
      search, first/last occurrence, rotated/ternary/interpolation search, find_peak.
    - Arrays: flatten (one-level and deep), chunk, remove_duplicates, moving_avarage,
      sliding_window, pad_array (in-place), rolling_sum, pairwise.
  - Section 2: aion.visualization
    - 1D: array, histogram, scatter, multiple arrays, mean, running mean, boxplot,
      density, CDF, error bars, rolling std, min-max band, autocorrelation, quantiles,
      scatter with fit, dual axis.
    - 2D: heatmap, confusion (raw and normalized), surface, contour, with values,
      correlation/similarity matrix, matrix histogram, masked heatmap, attention map,
      sparsity.
    - Training: history, single metric, train vs val, learning rate, metric with best,
      metrics grid, confidence band, early stopping, epoch time.
  - Section 3: What's new in v0.1.8 (aion.pdf)
    - get_documentation_statistics: module/function counts and per-module stats.
    - create_installation_guide: installation and setup guide (TXT or PDF).
    - create_quick_reference: compact function names by module (TXT or PDF).
    - validate_documentation: report which public functions lack docstrings.
    - create_documentation_index: INDEX.md listing all generated docs.

Run: python example.py
Requires: pip install aqwel-aion[full] (or at least matplotlib, numpy for visualization).
"""

import os
import copy

# ---------------------------------------------------------------------------
# 1. aion.algorithms — search and array utilities
# ---------------------------------------------------------------------------
# All functions below are from aion.algorithms; they use only the standard
# library (and type hints). Search functions expect sorted lists unless
# otherwise noted (e.g. linear_search, first/last occurrence).
print("=" * 60)
print("aion.algorithms — tests")
print("=" * 60)

from aion.algorithms import (
    binary_search,
    lower_bound,
    upper_bound,
    is_sorted,
    jump_search,
    find_peak_element,
    exponential_search,
    linear_search,
    First_Occurrence,
    Last_Occurrence,
    First_Last_Occurrence,
    roatated_search,
    ternary_search,
    interpolation_search,
    flatten_array,
    chunk_array,
    remove_duplicates,
    moving_avarage,
    flatten_deep,
    sliding_window,
    pad_array,
    rolling_sum,
    pairwise,
)

# --- Core search (sorted list required): index or None / insertion points ---
# binary_search: O(log n); returns index if target found, else None.
# lower_bound: first index where element >= target (e.g. insertion point for bisect).
# upper_bound: first index where element > target.
arr = [10, 20, 30, 40, 50, 60, 70]
print("binary_search(arr, 50):", binary_search(arr, 50))
print("binary_search(arr, 55):", binary_search(arr, 55))
print("lower_bound(arr, 35):", lower_bound(arr, 35))
print("upper_bound(arr, 50):", upper_bound(arr, 50))
# is_sorted: True if list is non-decreasing or non-increasing.
print("is_sorted([1,2,3]):", is_sorted([1, 2, 3]))
print("is_sorted([1,3,2]):", is_sorted([1, 3, 2]))

# --- Alternative search methods (sorted list): jump, exponential, linear ---
# jump_search: step through list then linear scan; good when step cost is low.
# exponential_search: double range then binary search; O(log n).
# find_peak_element: elements strictly greater than both neighbours.
slist = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
print("jump_search(slist, step=3, target=13):", jump_search(slist, step=3, target=13))
print("exponential_search(slist, 9):", exponential_search(slist, 9))
print("find_peak_element([1, 3, 2, 4, 1]):", find_peak_element([1, 3, 2, 4, 1]))

# --- First / last occurrence (return human-readable strings) ---
# linear_search and First_Occurrence/Last_Occurrence/First_Last_Occurrence
# work on any list; they return descriptive strings (or tuple of strings).
print("linear_search(slist, 7):", linear_search(slist, 7))
ulist_dup = [2, 4, 4, 4, 6, 8, 8, 10]
print("First_Occurrence(ulist_dup, 4):", First_Occurrence(ulist_dup, 4))
print("Last_Occurrence(ulist_dup, 4):", Last_Occurrence(ulist_dup, 4))
print("First_Last_Occurrence(ulist_dup, 8):", First_Last_Occurrence(ulist_dup, 8))

# --- Specialized search: rotated sorted array, ternary, interpolation ---
# roatated_search: list is a rotated sorted array (e.g. [4,5,6,7,1,2,3]).
# ternary_search: divides range into three parts; returns string.
# interpolation_search: suited to uniformly distributed data; returns string.
rotated = [4, 5, 6, 7, 1, 2, 3]
print("roatated_search(rotated, 2):", roatated_search(rotated, 2))
print("ternary_search([10,20,30,40,50], 30):", ternary_search([10, 20, 30, 40, 50], 30))
print("interpolation_search([10,20,30,40,50], 40):", interpolation_search([10, 20, 30, 40, 50], 40))

# --- Array utilities: flatten, chunk, dedupe, windowing, rolling, pairwise ---
# flatten_array: one-level flatten (list of lists -> list).
# flatten_deep: recursive flatten to any depth.
# chunk_array: consecutive chunks of given size; last chunk may be smaller.
# remove_duplicates: first occurrence kept, order preserved.
# moving_avarage: arithmetic mean of numeric list.
# sliding_window: generator of consecutive sublists of given size.
# rolling_sum: list of sums of each consecutive window (integers).
# pairwise: consecutive pairs as list of tuples (needs at least 2 elements).
print("flatten_array([[1,2],[3,4],[5]]):", flatten_array([[1, 2], [3, 4], [5]]))
print("flatten_deep([1, [2, [3, 4], 5], 6]):", flatten_deep([1, [2, [3, 4], 5], 6]))
print("chunk_array([1,2,3,4,5,6,7], size=3):", chunk_array([1, 2, 3, 4, 5, 6, 7], size=3))
print("remove_duplicates([3,1,2,1,4,2,3]):", remove_duplicates([3, 1, 2, 1, 4, 2, 3]))
print("moving_avarage([1.0, 2.0, 3.0, 4.0, 5.0]):", moving_avarage([1.0, 2.0, 3.0, 4.0, 5.0]))
print("list(sliding_window([1,2,3,4,5,6], 3)):", list(sliding_window([1, 2, 3, 4, 5, 6], 3)))
print("rolling_sum([1,2,3,4,5,6], 3):", rolling_sum([1, 2, 3, 4, 5, 6], 3))
print("pairwise([10, 20, 30, 40]):", pairwise([10, 20, 30, 40]))

# pad_array: extends list in-place to min_len by appending item. Use a copy if needed.
padded = copy.copy([1, 2, 3])
pad_array(padded, 6, 0)
print("pad_array([1,2,3], 6, 0) ->", padded)

print()

# ---------------------------------------------------------------------------
# 2. aion.visualization — all plot functions (saved to example_output/)
# ---------------------------------------------------------------------------
# Every plot function returns a matplotlib Figure. We save each to a file
# and close the figure to avoid "too many figures" warnings and keep the
# script suitable for headless/CI environments. Requires matplotlib (and
# numpy for some plots).
print("=" * 60)
print("aion.visualization — tests (plots saved to example_output/)")
print("=" * 60)

OUTPUT_DIR = "example_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

from aion.visualization import (
    plot_array,
    plot_histogram,
    plot_scatter,
    plot_multiple_arrays,
    plot_array_with_mean,
    plot_running_mean,
    plot_boxplot,
    plot_density,
    plot_cdf,
    plot_error_bars,
    plot_rolling_std,
    plot_min_max_band,
    plot_autocorrelation,
    plot_quantiles,
    plot_scatter_with_fit,
    plot_dual_axis,
    plot_matrix_heatmap,
    plot_confusion_matrix,
    plot_matrix_surface,
    plot_matrix_contour,
    plot_matrix_with_values,
    plot_correlation_matrix,
    plot_similarity_matrix,
    plot_matrix_histogram,
    plot_masked_heatmap,
    plot_confusion_matrix_normalized,
    plot_attention_map,
    plot_matrix_sparsity,
    plot_training_history,
    plot_metric,
    plot_train_vs_val,
    plot_learning_rate,
    plot_metric_with_best,
    plot_metrics_grid,
    plot_confidence_band,
    plot_early_stopping,
    plot_epoch_time,
)
from aion.visualization.utils import save_plot, close_figure


def save(name):
    return os.path.join(OUTPUT_DIR, name)


def save_and_close(fig, path):
    """Save figure to path and close it to free memory."""
    save_plot(fig, path)
    close_figure(fig)


# --- 1D array plots (index vs value, distributions, multi-series) ---
# 01–04: Basic line, histogram, scatter, multiple curves (e.g. loss vs val_loss).
save_and_close(plot_array([1, 3, 2, 5, 4], title="Array", show=False), save("01_plot_array.png"))
save_and_close(plot_histogram([1, 2, 2, 3, 3, 3, 4, 4, 4, 4], bins=4, title="Histogram", show=False), save("02_plot_histogram.png"))
save_and_close(plot_scatter([1, 2, 3, 4, 5], [5, 4, 3, 2, 1], title="Scatter", show=False), save("03_plot_scatter.png"))
save_and_close(
    plot_multiple_arrays(arrays=[[1, 2, 3, 4], [4, 3, 2, 1]], labels=["A", "B"], title="Multiple arrays", show=False),
    save("04_plot_multiple_arrays.png"),
)
# 05–06: Mean line overlay; running (moving) mean with window.
save_and_close(plot_array_with_mean([10, 12, 9, 11, 10, 13], title="Array with mean", show=False), save("05_plot_array_with_mean.png"))
save_and_close(
    plot_running_mean([15, 16, 14, 17, 18, 20, 19, 21, 22, 20, 18, 17], window_size=6, title="Running mean", show=False),
    save("06_plot_running_mean.png"),
)
# 07–09: Distribution views: boxplot, density curve, CDF.
save_and_close(plot_boxplot([1, 2, 2, 3, 3, 3, 4, 4, 5], title="Boxplot", show=False), save("07_plot_boxplot.png"))
save_and_close(plot_density([1.0, 1.2, 1.1, 2.0, 2.1, 2.2, 3.0], bins=5, title="Density", show=False), save("08_plot_density.png"))
save_and_close(plot_cdf([1, 2, 2, 3, 3, 4], title="CDF", show=False), save("09_plot_cdf.png"))
# 10–12: Uncertainty and rolling: error bars (x, y, yerr), rolling std, min-max band.
save_and_close(
    plot_error_bars([0, 1, 2, 3, 4], [1.0, 0.8, 0.6, 0.5, 0.4], [0.1, 0.12, 0.08, 0.07, 0.06], title="Error bars", show=False),
    save("10_plot_error_bars.png"),
)
save_and_close(plot_rolling_std([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], window_size=4, title="Rolling std", show=False), save("11_plot_rolling_std.png"))
save_and_close(plot_min_max_band([1, 2, 1.5, 3, 2.5, 4, 3.5], window_size=3, title="Min-max band", show=False), save("12_plot_min_max_band.png"))
# 13–16: Autocorrelation, quantiles (e.g. 0.25, 0.5, 0.75), scatter with linear fit, dual y-axis.
save_and_close(plot_autocorrelation([1, 2, 1, 2, 1, 2, 1], title="Autocorrelation", show=False), save("13_plot_autocorrelation.png"))
save_and_close(plot_quantiles([1, 2, 3, 4, 5, 6, 7, 8, 9], qs=[0.25, 0.5, 0.75], title="Quantiles", show=False), save("14_plot_quantiles.png"))
save_and_close(plot_scatter_with_fit([1, 2, 3, 4, 5], [2, 4, 5, 4, 6], title="Scatter with fit", show=False), save("15_plot_scatter_with_fit.png"))
save_and_close(
    plot_dual_axis([1, 2, 3, 4, 5], [1, 2, 3, 4, 5], [100, 80, 60, 40, 20], label1="Left", label2="Right", title="Dual axis", show=False),
    save("16_plot_dual_axis.png"),
)

# --- 2D matrix plots (heatmaps, confusion, surface, contour, derived matrices) ---
# 17–18: Generic heatmap; confusion matrix with optional class labels.
save_and_close(plot_matrix_heatmap([[1, 2, 3], [4, 5, 6], [7, 8, 9]], title="Matrix heatmap", show=False), save("17_plot_matrix_heatmap.png"))
save_and_close(
    plot_confusion_matrix([[50, 5], [8, 37]], labels=["Neg", "Pos"], title="Confusion matrix", show=False),
    save("18_plot_confusion_matrix.png"),
)
# 19–21: 3D surface, 2D contour, heatmap with cell value annotations.
save_and_close(plot_matrix_surface([[1, 2], [3, 4]], title="Matrix surface", show=False), save("19_plot_matrix_surface.png"))
save_and_close(plot_matrix_contour([[1, 2], [3, 4]], title="Matrix contour", show=False), save("20_plot_matrix_contour.png"))
save_and_close(plot_matrix_with_values([[1, 2], [3, 4]], title="Matrix with values", show=False), save("21_plot_matrix_with_values.png"))
# 22–23: Correlation matrix (rows = samples, cols = features); similarity matrix (cosine or dot).
save_and_close(
    plot_correlation_matrix([[1, 2], [2, 4], [3, 5], [4, 6]], labels=["F1", "F2"], title="Correlation matrix", show=False),
    save("22_plot_correlation_matrix.png"),
)
save_and_close(
    plot_similarity_matrix([[1, 0], [0.9, 0.1], [0, 1]], metric="cosine", title="Similarity matrix", show=False),
    save("23_plot_similarity_matrix.png"),
)
# 24–28: Histogram of matrix values; masked heatmap (True = hidden); normalized confusion; attention map; sparsity pattern.
save_and_close(plot_matrix_histogram([[1, 2], [3, 4]], bins=5, title="Matrix histogram", show=False), save("24_plot_matrix_histogram.png"))
save_and_close(
    plot_masked_heatmap([[1, 2], [3, 4]], [[False, True], [True, False]], title="Masked heatmap", show=False),
    save("25_plot_masked_heatmap.png"),
)
save_and_close(
    plot_confusion_matrix_normalized([[50, 5], [8, 37]], labels=["Neg", "Pos"], title="Confusion normalized", show=False),
    save("26_plot_confusion_matrix_normalized.png"),
)
save_and_close(plot_attention_map([[0.8, 0.2], [0.3, 0.7]], tokens=["A", "B"], title="Attention map", show=False), save("27_plot_attention_map.png"))
save_and_close(plot_matrix_sparsity([[1, 0, 0], [0, 1, 0], [0, 0, 1]], title="Matrix sparsity", show=False), save("28_plot_matrix_sparsity.png"))

# --- Training plots (metrics over epochs, learning rate, early stopping) ---
# history: dict of metric name -> list of values per epoch (e.g. from Keras/Torch).
history = {"loss": [1.0, 0.7, 0.5, 0.35, 0.25], "val_loss": [1.1, 0.8, 0.55, 0.4, 0.3], "accuracy": [0.5, 0.65, 0.75, 0.82, 0.88]}
# 29–31: All metrics on one axes; single metric; train vs validation curves.
save_and_close(plot_training_history(history, show=False), save("29_plot_training_history.png"))
save_and_close(plot_metric(history, "accuracy", title="Metric: accuracy", show=False), save("30_plot_metric.png"))
save_and_close(plot_train_vs_val(history["loss"], history["val_loss"], title="Train vs val", show=False), save("31_plot_train_vs_val.png"))
# 32–34: Learning rate schedule; one metric with best value highlighted; grid of all metrics.
save_and_close(plot_learning_rate([1e-3, 8e-4, 6e-4, 4e-4, 2e-4], title="Learning rate", show=False), save("32_plot_learning_rate.png"))
save_and_close(
    plot_metric_with_best(history, "val_loss", mode="min", title="Metric with best", show=False),
    save("33_plot_metric_with_best.png"),
)
save_and_close(plot_metrics_grid(history, cols=2, title="Metrics grid", show=False), save("34_plot_metrics_grid.png"))
# 35–37: Mean ± std band (e.g. across runs); early-stopping marker (patience, min/max); per-epoch duration.
save_and_close(
    plot_confidence_band([1.0, 0.7, 0.5, 0.35], [0.1, 0.08, 0.06, 0.05], title="Confidence band", show=False),
    save("35_plot_confidence_band.png"),
)
save_and_close(
    plot_early_stopping(history, "val_loss", patience=2, mode="min", title="Early stopping", show=False),
    save("36_plot_early_stopping.png"),
)
save_and_close(plot_epoch_time([12.1, 11.8, 11.9, 12.0, 11.7], title="Epoch time", show=False), save("37_plot_epoch_time.png"))

print("All algorithm tests completed.")
print("All visualization plots saved to", OUTPUT_DIR)

# ---------------------------------------------------------------------------
# 3. What's new in v0.1.8 — aion.pdf documentation helpers (for team visibility)
# ---------------------------------------------------------------------------
# New in this version: five functions for docs stats, installation guide,
# quick reference, docstring validation, and documentation index.
# Outputs go to example_output/ so other developers can see generated files.
print()
print("=" * 60)
print("What's new in v0.1.8 — aion.pdf documentation helpers")
print("=" * 60)

import aion
DOCS_DIR = os.path.join(OUTPUT_DIR, "v018_docs")
os.makedirs(DOCS_DIR, exist_ok=True)

# 3.1 get_documentation_statistics() — module/function counts and per-module stats
from aion.pdf import get_documentation_statistics
stats = get_documentation_statistics()
print("get_documentation_statistics():")
print("  module_count:", stats["module_count"])
print("  total_functions:", stats["total_functions"])
print("  sample modules:", [m["name"] for m in stats["modules"][:5]], "...")
if stats["modules_with_errors"]:
    print("  modules_with_errors:", stats["modules_with_errors"])

# 3.2 create_installation_guide() — installation and setup guide (TXT; PDF if ReportLab)
from aion.pdf import create_installation_guide
install_txt = os.path.join(DOCS_DIR, "aion_installation_guide.txt")
create_installation_guide(output_file=install_txt, format="txt")
print("create_installation_guide() ->", install_txt)

# 3.3 create_quick_reference() — compact function names by module (TXT; PDF if ReportLab)
from aion.pdf import create_quick_reference
quick_txt = os.path.join(DOCS_DIR, "aion_quick_reference.txt")
create_quick_reference(output_file=quick_txt, format="txt")
print("create_quick_reference() ->", quick_txt)

# 3.4 validate_documentation() — report which public functions lack docstrings
from aion.pdf import validate_documentation
validation = validate_documentation(module_name=None)
print("validate_documentation():")
print("  total_functions:", validation["summary"]["total_functions"])
print("  total_missing docstrings:", validation["summary"]["total_missing"])
if validation["missing_docstrings"]:
    sample = list(validation["missing_docstrings"].items())[:2]
    print("  sample missing:", dict(sample))

# 3.5 create_documentation_index() — INDEX.md listing all generated docs
from aion.pdf import create_documentation_index
index_path = create_documentation_index(output_dir=DOCS_DIR)
print("create_documentation_index() ->", index_path)

# Package metadata (v0.1.8): Author and Developer for team reference
print()
print("Package metadata (v0.1.8):")
print("  __version__:", aion.__version__)
print("  __author__:", aion.__author__)
print("  __developer__:", aion.__developer__)

print()
print("All v0.1.8 doc outputs saved to", DOCS_DIR)
print("Done.")
