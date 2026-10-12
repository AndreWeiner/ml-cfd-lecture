# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: 'Python 3.10.6 (''ml-cfd'': venv)'
#     language: python
#     name: python3
# ---

# %% [markdown]
# ---
# title: "Predicting the stability regime of rising bubbles — Exercises"
# jupyter: python3
# ---

# %% [markdown]
# [Download Python source](bubble_path_classification_exercise.py){.btn .btn-outline-primary .btn-sm data-noresolveinput="true" download="bubble_path_classification_exercise.py"}

# %% [markdown]
# ![CC](https://i.creativecommons.org/l/by/4.0/88x31.png)
#
# This work is licensed under a [Creative Commons Attribution 4.0 International License](http://creativecommons.org/licenses/by/4.0/).
#
# In this exercise, we refine the classification model built in the accompanying lecture. The model predicts the stability regime of rising bubbles based on Galilei and Eötvös numbers. Picking up from the final state of the lecture, we extract and add new data from experiments, split the data into training and validation sets, and investigate how the decision boundaries change as we keep optimizing the model's parameters.

# %% [markdown]
# ## Extracting additional data (optional)
#
# This section explains how to extract additional data from another related article. The [lecture repository](https://github.com/AndreWeiner/ml-cfd-lecture) also provides a download link to the full dataset.
#
# ### Installing Engauge Digitizer
#
# For this step, you need to download and install [Engauge Digitizer](http://markummitchell.github.io/engauge-digitizer/), or you use an alternative software of your choice with similar capabilities. The easiest way to install Engauge Digitizer on Ubuntu is via Snapcraft:
#
# ```
# sudo snap install engauge-digitizer
# ```
#
# ### Extracting the data
#
# The *datasets/path_shape_regimes/* folder contains a screenshot of the experimental data recorded and published in [Shapes and paths of an air bubble rising in quiescent liquids](https://aip.scitation.org/doi/10.1063/1.5006726) by D. M. Sharaf et al. (figure 4). [This video](https://www.youtube.com/watch?v=5ChKRaBs0ys) demonstrates how to extract data points for a single regime from the screenshot. These steps have to be repeated for all four regimes. Alternatively, the *datasets/path_shape_regimes/* folder already contains the extracted data. The new files follow the naming convection *exp_regime_[I,II,III,IV].csv*

# %% [markdown]
# ## Loading, processing and visualizing the data
#
# First, load the new experimental data in addition to the numerical data used in the lecture. The lecture notebook shows how to fetch the data with the Pandas library. When creating the *regime* column, merge regimes *IV* and *V* of the numerical data (set the value for both regimes to *IV*). Create also ordinal categories for each regime as demonstrated in the lecture. To verify that the data were loaded correctly, sample a few rows or use the `DataFrame.describe()` method.
#
# Since the investigated Galilei and Eötvös numbers are not very homogeneously distributed, we work with the common logarithm of both features. An additional normalization to the value range $[-1, 1]$ is not strictly necessary but nonetheless a good practice. To see the effect of the logarithmic scaling, plot the original and the scaled regime data as scatter plots.

# %% [markdown]
# ## Training a neural network-based classification model
#
# ### Training and validation data
#
# With the additional data, we can allocate more feature-label-pairs for the validation set. As in the lecture, the training data are used to optimize the model's free parameters, and the validation data help to monitor the model's generalization. Create a new `TensorDataset` with $\tilde{Ga}$ and $\tilde{Eo}$ as features and the ordinal regime representation as label (ordinal means that the labels are 0, 1, 2, 3, 4 rather than "I", "II", ...). Split the dataset using `random_split` into two sets holding roughly $80\%$ and $20\%$ of all samples. Finally, create `Dataloader`s with the batch size being the full length of the training and validation sets, respectively.
#
# ### Creating and optimizing the regime model
#
# The following steps guide you through creating, training, and evaluating the regime model:
#
# - create a new fully-connected neural network with a suitable number of input and output neurons
# - train the model for $200$ epochs using the generic training loop, the `AdamW` optimizer, and a constant learning rate of $\lambda = 0.001$
# - plot training and validation losses
# - create a plot showing the original data points and the decision boundaries predicted by the model (last figure in the lecture notebook)
# - re-initialize the model, train for $\{ 200, 500, 1000, 2000\}$ epochs, monitor training and validation loss, and observe how the decision boundaries change
#
# **Congratulations! This completes the sixth exercise session.**

# %% [markdown]
#
