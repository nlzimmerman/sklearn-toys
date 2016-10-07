# Sklearn tools I wrote

Notebooks and scratch code are in the `sandbox` directory. Ignore them.

Scikit-learn addons are in the `sklearn_addons` directory.

## Logistic imputation

Right now, there's just one thing there: `preprocessing/logistic_imputation.py`.
This implements a class analogous to sklearn's `Imputer` class, but instead of using the mean, median, or most frequent values of a feature (column), it uses a logistic regression to fill in the missing data. This only works for categorical data!

For each feature, a logistic model is train where the target vector is the part of this feature vector that is _not_ missing, and the training matrix all the other features from the same set of samples, where data being missing _is_ treated as a feature, i.e. if the data had two categories before, there will be three used to train the model. However since missing data is excluded from the target, only two categories will be predicted.

Concretely, for a matrix whose values can be 0 or 1, with missing data denoted as X, e.g.

```
[[ X 0 X 1 ]
 [ 1 1 0 X ]
 [ 0 X 1 0 ]]
```
the target for the first column will be `[1, 0]` and the training matrix will be
```
[[ 1 0 X ]
 [ X 1 0 ]]
```
which will be transformed through one-hot encoding to
```
[[ 1 0 0 0 1 0 0 0 1 ]
 [ 0 0 1 1 0 0 0 1 0 ]]
```
One these models are trained, they will be used to fill in the missing values.

To facilitate cross-validation, a `score()` method is implemented which combines the scores of each logistic model.
