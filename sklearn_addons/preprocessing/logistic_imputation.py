# coding=utf-8


# Following the patterns set in sklearn/preprocessing/imputation.py as much as possible.

from sklearn import warnings

import numpy as np
import numpy.ma as ma
from scipy import sparse
from scipy import stats

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils import check_array
from sklearn.utils.fixes import astype

from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression


import six
import numbers



zip = six.moves.zip
map = six.moves.map

__all__ = [
  'LogisticImputer'
]

# This is directly, directly taken from skelarn.preprocessing.imputation
# as of the master on github on the day I made this file.
# I have copy/pasted it instead of importing it to avoid importing a private.

# Ugh, four spaces per newline. :)
# def _get_mask(X, value_to_mask):
#     """Compute the boolean mask X == missing_values."""
#     if value_to_mask == "NaN" or np.isnan(value_to_mask):
#         return np.isnan(X)
#     else:
#         return X == value_to_mask

# End part that I 100% did not write at all.

def _knockout_index(length, knockout):
  """Convenience function that returns list(range(length)), with knockout removed."""
  return list(range(0,knockout)) + list(range(knockout+1, length))

class LogisticImputer(BaseEstimator, TransformerMixin):
  """Logistic imputation transformer for completing missing values.

  Parameters
  ----------
  missing_values : the 0 integer, "NaN", or numpy.nan, optional (default = "NaN")
    The placeholder for the missing values. All occurrences of `missing_values`
    will be imputed. Note that "NaN" is a convenience reference to np.nan.
    Any integer other than 0 will return an error!
    The **only** valid encoding schemes are
    - 0 for missing data and [1, 2, 3, …] for valid data or
    - np.nan for missing data and [0, 1, 2, 3, …] for valid data.
  C : float, optional (default = 1)
    The regularization parameter to pass to sklearn.linear_model.LogisticRegression
  n_values : int or "auto". The number of distinct, **non-missing** values in the data.
    If "auto", this will be found by inspection of the traning matrix.
  """
  def __init__(self, missing_values = "NaN", C = 1, n_values = "auto"):
    self.missing_values = missing_values
    self.C = C
    self.n_values = n_values

  def _nan_to_placeholder_int(self,X):
    # If the missing value is np.nan, replace it with a value that the OneHotEncoder can handle,
    # specifically, the largest value currently present plus one.
    if np.isnan(self.missing_values_):
      np.place(X, np.isnan(X), [self.n_values_])
      return (X, self.n_values_)
    else:
      return (X, 0)

  def _generate_onehot_encoder(self):
    return OneHotEncoder(n_values=int(self.n_values_+1))

  def fit(self, X, y=None):
    """Fit the imputer on X.

    Parameters
    ----------
    X : {array-like, sparse matrix}, shape (n_samples, n_features)
        Input data, where ``n_samples`` is the number of samples and ``n_features``
        is the number of features.

    Returns
    -------
    self : object
        Returns self.
    """

    # Note: the axis argument from LogisticImputer is _definitely not_ applicable
    # here.

    if isinstance(self.missing_values, six.string_types) and self.missing_values.lower() == "nan":
      self.missing_values_ = np.nan
    elif isinstance(self.missing_values, numbers.Number) and self.missing_values == 0 or np.isnan(self.missing_values):
      self.missing_values_ = self.missing_values
    else:
      raise Exception("Don't understand missing value {}".format(self.missing_values))

    # only implementing a dense fit for now.
    X = check_array(X, accept_sparse ='csc', dtype=np.float64, force_all_finite=False)
    if sparse.issparse(X):
      X = X.todense()
    else:
      # if it's already dense, we still want to make a copy to avoid overwriting the original.
      X = X.copy()

    # we have to cast to float because np.int* doesn't support nan, but let's additionally
    # make sure that everything is an int.
    _intchecker = np.vectorize(lambda x: np.isnan(x) or x==int(x))
    if not np.all(_intchecker(X)):
      raise Exception("Matrix appears not to be int and nan.")

    n_features = X.shape[1]
    self.n_features_ = n_features
    # So… in order for OneHotEncoder to work, we have to turn the missing values into
    # a scalar. So we'll do that here.
    # This makes the exact same assumptions as preprocessing.OneHotEncoder — it assumes
    # that data are integers in the range [0, 1, .. N].
    if self.n_values == "auto":
      # Again: the __only__ legal values of self.missing_values_ are np.nan and 0
      # This simplifies the logic considerably.
      if np.isnan(self.missing_values_):
        # [nan, 0, 1, 2] = 3 values
        self.n_values_ = np.nanmax(X) + 1
      elif self.missing_values_ == 0:
        # [0, 1, 2] = 2 values because 0 denotes missing value
        self.n_values_ = np.max(X)
      else:
        raise Exception("Missing value {} seems not to be 0 or nan, and a previous check didn't catch this?".format(self.missing_values_))
    else:
      self.n_values_ = self.n_values

    # where m is the missing value now that we've replaced the nans.
    X, m = self._nan_to_placeholder_int(X)

    # self.n_values_ does not include the missing value, but it counts as a value for the one-hot encoder.
    # self.one_hot_encoder_ = OneHotEncoder(n_values=int(self.n_values_+1))
    # self.one_hot_encoder_.fit(X)

    feature_predictors = list()
    for feature_idx in range(n_features):
      y_with_missing = np.ravel(X[:, feature_idx])
      # There may be a more numpy-like way of doing this, but what I'm doing here
      # is just getting the row indices of the parts of the target that don't
      # correspond to missing data. In the case that missing data is zero, I just
      # used
      # [nonzero_rows] = y_with_missing.nonzero()
      # but that doesn't work when the missing data is a large number
      # because I translated that over from np.nan
      nonmissing_row_indices = [i for x, i in zip(y_with_missing, range(len(y_with_missing))) if x != m]
      # There may be a more efficient way of slicing this? I have not really
      # performance tested this code.
      # x' is this matrix, minus the feature column, minus the columns where the feature column is undefined.
      x_prime = X[:, _knockout_index(n_features, feature_idx)][nonmissing_row_indices,:]
      x_prime_onehot = self._generate_onehot_encoder().fit_transform(x_prime)
      # y is the feature vector with the missing values removed.
      y = y_with_missing[nonmissing_row_indices]
      logreg = LogisticRegression(C = self.C)
      logreg.fit(x_prime_onehot, y)
      feature_predictors.append(logreg)

    self.feature_predictors_ = feature_predictors

  def transform(self, X):
    """Impute all missing values in X.
    """

    check_is_fitted(self, 'feature_predictors_')

    if sparse.issparse(X):
      X = X.todense()
    else:
      # I prefer not to overwrite the original.
      # Ideally this would be a flag you could pass.
      X = X.copy()
    # This is where we will write the fitted variables to.
    # It was either make a copy, or do two sweeps, since imputed data shouldn't be
    # used as a feature to impute other data.
    X_fit = X.copy()
    n_features = X.shape[1]
    if n_features != self.n_features_:
      raise Exception("Number of features does not match fit data!")
    X, m = self._nan_to_placeholder_int(X)

    for feature_idx in range(n_features):
      # we are only doing this slice to find the missing value indices. No training here!
      y_with_missing = np.ravel(X[:, feature_idx])
      # As above, but now I'm finding the indices where data _is_ missing so I can replace it.
      missing_row_indices = [i for x, i in zip(y_with_missing, range(len(y_with_missing))) if x == m]
      x_prime = X[:, _knockout_index(n_features, feature_idx)][missing_row_indices,:]
      x_prime_onehot = self._generate_onehot_encoder().fit_transform(x_prime)
      y_fit = self.feature_predictors_[feature_idx].predict(x_prime_onehot)
      X_fit[missing_row_indices, feature_idx] = y_fit

    return X_fit

  def score(self, X, y=None, normalize=True, sample_weight=None, return_normalization_basis=False):
    if sample_weight is not None:
      raise Exception("We don't know how to score unevenly-weighted samples.")
    # Hopefully this is self-evident.
    # This is the same workflow as above, only now we are predicting the values we already know,
    # so that we can compare them with the truth. The is NOT the same as self.transform(),
    # which never changes a known value.
    check_is_fitted(self, 'feature_predictors_')
    if sparse.issparse(X):
      X = X.todense()
    else:
      # I prefer not to overwrite the original.
      # Ideally this would be a flag you could pass.
      X = X.copy()
    X_fit = X.copy()
    n_features = X.shape[1]
    if n_features != self.n_features_:
      raise Exception("Number of features does not match fit data!")
    X, m = self._nan_to_placeholder_int(X)
    for feature_idx in range(n_features):
      # we are only doing this slice to find the missing value indices. No training here!
      y_with_missing = np.ravel(X[:, feature_idx])
      # As above, but now I'm finding the indices where data _is_ missing so I can replace it.
      not_missing_row_indices = [i for x, i in zip(y_with_missing, range(len(y_with_missing))) if x != m]
      # Again: here, we are predicting over the thing that we already know.
      # You're using cross-validated data, right? :)
      x_prime = X[:, _knockout_index(n_features, feature_idx)][not_missing_row_indices,:]
      x_prime_onehot = self._generate_onehot_encoder().fit_transform(x_prime)
      y_fit = self.feature_predictors_[feature_idx].predict(x_prime_onehot)
      X_fit[not_missing_row_indices, feature_idx] = y_fit

    # So now we have X and X_fit, and the nans in X and X_fit are in the same places.
    if np.isnan(self.missing_values_):
      missing_check = lambda x: np.isnan(x)
    else:
      missing_check = lambda x: x==self.missing_values_
    # In X, we did an inline replace of nan with an integer after we copied it to make X_fit,
    # so the original nan value will only be present in Y.
    # Alternatively, we could check for x==m, but we're taking it on faith
    # That they will be the same (and so we aren't throwing a potential exception.)
    X_notmissing, X_fit_notmissing = zip(
      *[
        (x,y) for x,y in zip(np.ravel(X), np.ravel(X_fit)) if not (missing_check(y))
      ]
    )
    if not return_normalization_basis:
      return accuracy_score(X_notmissing, X_fit_notmissing, normalize, sample_weight)
    else:
      return (accuracy_score(X_notmissing, X_fit_notmissing, normalize, sample_weight), len(X_notmissing))
