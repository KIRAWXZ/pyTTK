# pyTTK
Python implementation of transductive top k algorithm proposed by Liu, et al (2015) in [arXiv:1510.05976](https://arxiv.org/abs/1510.05976).

## The Model
The model is a modification of a linear support vector machine restricted to predict at most k examples of a known test set as positive examples (making it a transductive method) in order to optimize for precision@k rather than overall accuracy. This sort of optimization is relevant to many policy settings where the set of entities for a prediction is known in advance and limited resources dictate that only a set number of them can be subjected to some treatment or intervention and performance of the model is most important at the top of the list, for instance, targeting health inspections to restaurants at highest risk for violations or identifying police officers most at risk of an adverse event.

See [Liu, et al (2015)](https://arxiv.org/abs/1510.05976) for details, but in short, TTK optimizes the typical SVM hinge loss (with L2 regularization of the `w` vector) subject to the constraint that at most k test examples fall above the decision boundary. Note that in some cases (such as relatively rare events) this algorithm may return a result with fewer than k examples predicted as positives.

## Usage
pyTTK requires `numpy`, `pandas`, and `sklearn` (see requirements.txt). To use, simply clone the repository and create directories `logs/` and `pickles/` to hold logs and model pickles, respectively.

There are two ways to use the model: with a command line interface or as an object with an sklearn-style interface:

### command line interface
For the command line interface, training and test data are expected to be in CSV or HDF5 format and contain a header row. Labels are expected to be in a column named `outcome` and columns named `entity_id` and `as_of_date` will be ignored from the feature set.

The TTK model can be run on the command line with `python ttk.py` using the following parameters:
* `--trainmat`: Required location of the training data file, in CSV or HDF5 format with a header row.
* `--testmat`: Required location of the test data file, in CSV or HDF5 format with a header row.
* `--k`: Top k number of test examples to optimized to (as an integer); either `--k` or `--kfrac` is required.
* `--kfrac`: Top fraction of the test examples to optimized to (as a fraction); either `--k` or `--kfrac` is required.
* `--C`: Slack penalty parameter (defaults to 1.0)
* `--maxiter`: Maximum number of iterations for the subgradient descent (defaults to 5000)
* `--scaledata`: Method for scaling data: `z` will scale to mean 0, stdev 1 (based on training examples); `minmax` will scale to range [0,1] (based on min & max in training set); `none` (the default) will not scale the input data
* `--datatype`: Parameter to indicate how input data is stored: `csv`, `hdf5`, or `infer` (the default) from file extension
* `--verbose`: Debug-level logging

For example to run the model for precision@300 with a penalty of 1.0 and min-max data scaling,

```python ttk.py --trainmat my_training_data.csv --testmat my_test_data.csv --k 300 --C 1.0 --scaledata minmax```

### sklearn-style interface
The `ttklearn.TTKClassifier` class provides a simple wrapper for the TTK model with an interface similar to `sklearn` models. Initialize an instance of the classifier with values for `k`, `C`, and (optionally) `max_iter`. The model can then be fit on training data (and test features) using `fit(x_train, y_train, x_test)` (additional `init_w` and `init_b` parameters can optionally be passed with a starting point for the optimization, otherwise `sklearn.svm.LinearSVC` will be used).

Classification predictions on a test set can be obtained with `predict(x_test)` and the decision function values with `decision_function(x_test)`. Additionally, call `test_precision(x_test, y_test)` to calculate precision@k for the model.

