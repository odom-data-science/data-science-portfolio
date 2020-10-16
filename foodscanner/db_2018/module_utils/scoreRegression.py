import numpy as np
import pandas as pd


def scoreClassif(y_true, y_pred):
    """Computes a classification score by setting the threshold of y_true, then
    determines whether the learning algorithm predicted well or not (0 or 1) (with y_pred)

    Parameters
    ----------
    y_true : array-like
        y is a list or an array  that contain the target variable.
    y_pred : array-like
        Result of model.predict(x)

    Returns
    -------
    score_classifier (loss) : float or ndarray of floats
        Computes the sum of good prediction divided by the product of y_pred.shape"""
    threshold = np.percentile(y_true, 50, axis=0)
    y_true_classif = y_true < threshold
    y_pred_classif = y_pred < threshold
    score_classifier = np.sum(
        y_pred_classif == y_true_classif) / np.prod(y_pred.shape)
    return score_classifier


def scoreClassifier(model, x, y):
    """Computes classification score

    Parameters
    ----------
    model (object) : The object to predict the  data. Compute y_pred of the model
    x : array-like (0 or 1)
        x is a list or an array  that contain the independent variables.
    y : array-like (0 or 1)
        y is a list or an array  that contain the dependent variables.

    Return
    ------
    score_classifier (loss) : float or ndarray of floats
        Computes the sum of good prediction divided by the product of y_pred.shape"""
    # On reshape y_pred pour le mettre dans la même shape que y_true pour pouvoir les comparer
    y_pred = model.predict(x)
    y_pred = y_pred.reshape(y.shape[0], y.shape[1])
    score_classifier = np.sum(y_pred == y) / np.prod(y_pred.shape)
    return score_classifier

# Pre-processing


def one_hot_encoder(df, column_list):
    """Takes in a dataframe and a list of columns
    for pre-processing via one hot encoding"""
    df_to_encode = df[column_list]
    df = pd.get_dummies(df_to_encode)
