# Data-source documentation for the classification exercise

The exercise is `notebooks/bubble_path_classification_exercise.py`.
The supplied observations are in the adjacent `tripathi2015_digitization/` and
`sharaf2017_digitization/` directories. Their README files describe the sources
and extraction procedures; the exercise loads both CSVs directly.

`source_card_template.md` provides the data-source documentation template for
checking definitions, source conditions, original categories, chosen common
class names, exclusions, and pooling limitations. The legacy
`additional_points_template.csv` is available for an optional further source;
it is not needed to load the supplied Sharaf dataset.

Students investigate label compatibility and choose their mapping before merging
and splitting the observations. The supplied NumPy helper supports the chosen
number of output classes and exports paired learning-curve figures showing
validation loss and accuracy as the training subsets grow.
