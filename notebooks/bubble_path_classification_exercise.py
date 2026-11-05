# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "Surrogate models for discrete predictions"
# subtitle: "Exercise — extending a bubble-regime dataset"
# jupyter: python3
# bibliography: ../references.bib
# execute:
#   eval: false
# ---
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](https://creativecommons.org/licenses/by/4.0/).
#
# ## Aim of the exercise
#
# The [lecture](bubble_path_classification.py) developed a classifier from digitized numerical observations. The central question of this exercise is: **How much can we improve the robustness and accuracy of this classifier based on additional numerical or experimental data?**
#
# Investigate the lecture data from @tripathi2015 alongside experimental observations from @sharaf2017. Inspect the two sources, decide how to represent their labels consistently, and examine how validation performance changes as the training dataset grows.
#
# ## Files and preparation
#
# [Download exercise source](bubble_path_classification_exercise.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="bubble_path_classification_exercise.py"}
# [Download supplied helpers](classification_exercise_tools.py){.btn .btn-outline-primary .btn-sm download="classification_exercise_tools.py"}
# [Download Tripathi data](../datasets/path_shape_regimes/tripathi2015_digitization/points.csv){.btn .btn-outline-primary .btn-sm download="tripathi2015-points.csv"}
# [Download Sharaf data](../datasets/path_shape_regimes/sharaf2017_digitization/points.csv){.btn .btn-outline-primary .btn-sm download="sharaf2017-points.csv"}
#
# The helpers also use [figure_style.py](figure_style.py) and [figure_utils.py](figure_utils.py). Outside a repository clone, place these three helper files beside the exercise source and save the two CSVs as `tripathi2015-points.csv` and `sharaf2017-points.csv`. The code runs locally in Python, not on this HTML page. Run the code blocks in sequence in a Python editor, or open the source as a notebook using Jupytext. NumPy and Matplotlib are needed; figure exports use LaTeX by default. An explicit `use_tex=False` option is available when TeX is unavailable.
#
# ### Getting started with the supplied code
#
# **The supplied code is our starting point for experimenting: we do not need to implement a classifier or training loop from scratch.** The main aim is to run comparisons, try different configurations, and interpret what changes when we vary the data, training-set size, or network architecture.
#
# 1. Download the exercise source, the three helper files linked above, and both CSV files into one working folder. Use the filenames `tripathi2015-points.csv` and `sharaf2017-points.csv` for the data.
# 2. Run Python from that folder. With these filenames, the supplied `dataset_file` function finds the data automatically. If the CSVs are stored elsewhere, replace the paths in its `choices` list with the paths to your local files.
# 3. Run the loading block below first and inspect its printed summary. Before training, complete `label_maps` and `class_names` in “A common classification problem” after comparing the source labels. These are the intended data-definition choices; the remaining training and plotting code can be used as supplied.
# 4. Run the remaining blocks in order to obtain a baseline. Then change one setting at a time, such as the training fractions, network widths and depths, or learning rate, and compare the results using the same validation cases. Keep a record of the settings and explain the observed differences.
#
# ## Load and investigate the two datasets
#
# The lecture dataset contains 104 digitized markers from Figure 1 of @tripathi2015. The additional CSV contains 183 discernible markers from the experimental Figure 4 of @sharaf2017. Their positions were extracted from the embedded high-resolution image and calibrated using the logarithmic axis ticks. Touching and partly obscured markers were reviewed visually. These are approximate digitized coordinates, not the authors' original raw observations; repeated experiments and fully hidden markers cannot be recovered from the map.
#
# The [Tripathi extraction notes](../datasets/path_shape_regimes/tripathi2015_digitization/README.md) and [Sharaf extraction notes](../datasets/path_shape_regimes/sharaf2017_digitization/README.md) document the sources and methods. The [Sharaf extraction overlay](../datasets/path_shape_regimes/sharaf2017_digitization/extraction-overlay.png) allows the recovered positions to be checked against the publication.
#
# - Load both files and inspect their column names, parameter ranges, and class counts.
# - Plot the two datasets using logarithmic Ga and Eo axes. Make the sources distinguishable and retain their original labels initially.
# - Compare the marker legends and descriptions in the two papers. Do the same category identifiers describe the same physical behavior? Can every source label be translated unambiguously into the lecture's classes?
# - Check the definitions of Ga and Eo, the length scale, and the density and viscosity ratios. Which differences might affect the comparison even after the labels are reconciled?
#

# %%
from pathlib import Path
from collections import Counter
import csv
import json
import numpy as np
from classification_exercise_tools import (
    split_by_source_and_class, learning_curve, save_learning_curve,
    compare_architectures, save_architecture_comparison,
)


def dataset_file(study):
    choices = [Path(f"{study}-points.csv"),
               Path(f"../datasets/path_shape_regimes/{study}_digitization/points.csv"),
               Path(f"datasets/path_shape_regimes/{study}_digitization/points.csv")]
    return next(p for p in choices if p.is_file())


def read_source(study, label_column):
    with dataset_file(study).open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["Ga"], row["Eo"] = float(row["Ga"]), float(row["Eo"])
        row["source_id"] = study
        row["source_regime"] = row[label_column]
        row["group_id"] = row["point_id"]
    return rows


tripathi = read_source("tripathi2015", "regime")
sharaf = read_source("sharaf2017", "source_regime")
for name, rows in [("Tripathi", tripathi), ("Sharaf", sharaf)]:
    print(name, len(rows), Counter(r["source_regime"] for r in rows))
    print("Columns:", list(rows[0]))
    print("Ga range:", min(r["Ga"] for r in rows), max(r["Ga"] for r in rows))
    print("Eo range:", min(r["Eo"] for r in rows), max(r["Eo"] for r in rows))
    print(rows[:2])

# %% [markdown]
#
# ## A common classification problem
#
# Before pooling the observations, define a common prediction target. Record and justify the chosen definition in the [data-source documentation template](../datasets/path_shape_regimes/classification_exercise/source_card_template.md).
#
# - Explain any differences between the source classifications. Propose a set of classes that can be supported by both datasets and justify any aggregation or exclusion.
# - Complete the two mapping dictionaries below. Each original label maps to a chosen common class name, or to `None` if those observations must be excluded. Retain the original labels and document the reasons for exclusions. No new subtype should be inferred solely from a marker's location.
# - Compare the retained class counts and parameter coverage. Is the resulting prediction task still exactly the same as in the lecture?
#
# Matching definitions and labels does not establish that Ga and Eo alone explain both sources. Density and viscosity ratios, release conditions, or other unmodeled differences may cause conflicting observations. Keep the source IDs so that such differences remain visible.
#

# %%
# Complete these after investigating the source descriptions.
label_maps = {"tripathi2015": None, "sharaf2017": None}
class_names = None  # Ordered list of the chosen common class names.

if class_names is None or any(v is None for v in label_maps.values()):
    raise ValueError("Choose and document the common classes and both label mappings first")
if len(class_names) < 2 or len(set(class_names)) != len(class_names):
    raise ValueError("At least two distinct common classes are needed")

merged, excluded = [], []
for row in tripathi + sharaf:
    mapping = label_maps[row["source_id"]]
    common_label = mapping[row["source_regime"]]
    if common_label is None:
        excluded.append(dict(row))
    elif common_label in class_names:
        merged.append(dict(row, regime=common_label))
    else:
        raise ValueError("A mapped label is missing from class_names")
print("Retained:", Counter((r["source_id"], r["regime"]) for r in merged))
print("Excluded:", len(excluded))

# %% [markdown]
#
# ## Merge, then split into training and validation data
#
# Split the harmonized, merged dataset once into approximately 80% training and 20% validation observations. The split preserves representation by source and common class, and related observations remain in the same partition. There is no test set. Each CSV row represents a recoverable marker; the original repeated experiments cannot be reconstructed from the published images.
#
# - Run the split below and inspect class and source counts in both partitions. Confirm that no source/case group crosses the split.
# - Keep the validation cases unchanged throughout the training-size comparison. Adding validation cases as the training set grows would change the quantity being compared.
#

# %%
training, validation = split_by_source_and_class(merged, seed=42)
training_groups = {(merged[i]["source_id"], merged[i]["group_id"]) for i in training}
validation_groups = {(merged[i]["source_id"], merged[i]["group_id"]) for i in validation}
assert not training_groups & validation_groups
for name, indices in [("Training", training), ("Validation", validation)]:
    print(name, len(indices), Counter((merged[i]["source_id"], merged[i]["regime"])
                                     for i in indices))

# %% [markdown]
#
# ## Validation performance as the training dataset grows
#
# Train the same MLP on increasingly large, nested subsets of the training pool: approximately 20%, 40%, 60%, 80%, and 100%. Subsets retain representation by source and common class; their actual sizes depend on rounding and case grouping. For each random seed, all observations in a smaller subset also occur in the larger subsets. Use a smaller network than the lecture's two-hidden-layer demonstration to keep the repeated fits inexpensive: one hidden layer with 16 tanh units and one softmax output per chosen common class. The classification model, loss, and preprocessing follow the same principles. In the code, common class names are represented by zero-based indices `0,...,K-1`; these are identifiers, corresponding to the lecture's mathematical indices `1,...,K`.
#
# Logarithmic min–max scaling is fitted separately on each training subset. The corresponding transformation is reused for the fixed validation set; validation observations never contribute to the scaling or weight updates. Use full-batch Adam with learning rate 0.01, at most 500 epochs, and early-stopping patience 100. Restore the best-validation weights. Repeating the experiment with three seeds shows sensitivity to subset selection and initialization; these repetitions are not independent validation datasets or confidence intervals.
#
# - Run the training-size comparison and plot validation cross entropy against the actual number of training points. Also inspect validation accuracy.
# - Does validation loss decrease as the number of training points increases? Is that trend monotonic and consistent across seeds?
# - Does accuracy tell the same story as cross entropy? Where might differences between the physical investigations limit the benefit of additional data?
# - Does this experiment isolate the benefit of Sharaf's data specifically, or the effect of training size in the pooled dataset? Explain the distinction.
#

# %%
configuration = dict(width=16, learning_rate=0.01, epochs=500, patience=100)
results = learning_curve(merged, training, validation, class_names,
                         fractions=(.2, .4, .6, .8, 1.), seeds=(11, 22, 33),
                         **configuration)
output = Path("output/bubble-classification")
paths = save_learning_curve(output, results, use_tex=True)
for result in results:
    print(result["seed"], result["training_samples"],
          result["validation_cross_entropy"], result["validation_accuracy"])
print("Figures:", paths)

# %% [markdown]
#
# The generated `learning-curve-light.svg` and `learning-curve-dark.svg` show validation cross entropy and accuracy against training size. A falling loss would support improved probability predictions on this fixed validation subset; improvement is not guaranteed. The validation set selects early-stopping checkpoints and is reused across runs, so its score is not an independent final test score. It also does not establish robustness to an unseen fluid system or investigation.
#
# ## Network width, depth, and overfitting
#
# Keep the entire training pool, the validation cases, and the training-fitted scaler fixed. Vary the number of hidden units and hidden layers: `(8,)`, `(16,)`, and `(32,)` describe one hidden layer; `(16,16)` and `(16,16,16)` add one or two more hidden layers. Also test two hidden layers with 32 neurons each. All hidden layers use tanh, and the softmax output still has one unit per common class.
#
# For this diagnostic comparison, all models train for the same 2,000 epochs without a patience-based stopping trigger. Retain the complete training and validation loss histories, the parameters at the validation minimum, and the final parameters. Use these records to examine what happens after validation performance stops improving. The final model and validation-selected model are reported separately.
#
# - Compare the networks using the same learning rate, split, and initialization seeds. Record the number of trainable parameters alongside width and depth.
# - Plot validation accuracy for both the selected and final models. Does a larger network consistently improve accuracy across the seeds?
# - Inspect the training and validation loss histories. Where does training loss continue to decrease while validation loss increases? Can validation accuracy stay unchanged while validation cross entropy worsens?
# - Compare each validation minimum with the final epoch. How would early stopping change the parameters retained? Distinguish possible overfitting from underfitting or incomplete optimization; a larger network is not automatically an overfitted network.
#

# %%
architecture_results = compare_architectures(
    merged, training, validation, class_names,
    architectures=((8,), (16,), (32,), (16,16), (16,16,16), (32,32)),
    seeds=(11, 22, 33), epochs=2000, learning_rate=0.01,
)
architecture_paths = save_architecture_comparison(output, architecture_results, use_tex=True)
for run in architecture_results:
    print(run["hidden_sizes"], run["seed"], run["parameters"],
          run["best_validation_accuracy"], run["final_validation_accuracy"],
          run["best_epoch"], run["final_epoch"])
print("Architecture figures:", architecture_paths)

# %% [markdown]
#
# The accuracy figure compares the same validation observations across architectures. The loss-history figure shows seed 11 for readability; all three seeds are retained in the JSON output. Dots identify the validation minima. The range of outcomes across seeds is not a confidence interval. Selecting architectures against the same validation set is model selection, not an independent assessment of a final model's predictive accuracy.
#
# ## Document and interpret the comparison
#
# Retain the source labels, common label mapping, exclusions, split, training settings, learning-curve results, and architecture comparisons so that the comparison can be reproduced.
#
# - Save the merged observations, training and validation IDs, mappings, and excluded observations.
# - Include the data-source documentation and paired learning-curve, accuracy, and loss-history figures.
# - Explain the main trend, its sensitivity to the seed, and at least one physical or statistical limitation. A result showing little improvement or deterioration is informative too.
#

# %%
record = dict(label_maps=label_maps, class_names=class_names, configuration=configuration,
              split_seed=42, merged=merged, excluded=excluded,
              training_indices=training.tolist(), validation_indices=validation.tolist(),
              results=results, architecture_results=architecture_results)
(output / "comparison.json").write_text(json.dumps(record, indent=2) + "\n")

# %% [markdown]
#
# ### Optional numerical data source
#
# The numerical observations of @canolozano2016 may also be considered. Their definitions and labels would require the same compatibility check before inclusion; this source is not needed for the comparison above.
#
# ## References
#
# ::: {#refs}
# :::
