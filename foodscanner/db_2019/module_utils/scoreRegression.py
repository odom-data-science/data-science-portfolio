import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


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
    y_true_classif = y_true_classif.flatten()
    y_pred_classif = y_pred < threshold
    y_pred_classif = y_pred_classif.flatten()
    score_classifier = np.sum(
        y_pred_classif == y_true_classif) / np.prod(y_pred.shape)
    return score_classifier


def scoreClassifier(model, x, y):  # do not use it
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

# -----------------------------------------
# ---------- ONE HOT ENCODING ----------
# -----------------------------------------
# GET DUMMIES


def one_hot_encoder(df, column_list):
    """Takes in a dataframe and a list of columns
    for pre-processing via one hot encoding"""
    df_to_encode = df[column_list]
    df = pd.get_dummies(df_to_encode)

# ONE HOT ENCODER


def onehot_encoder(values, index_df, columns_df):
    """
    Create a dataframe with value of a one-hot-encoded column: unique categorical variables are transformed into columns

    Parameters
    ----------
    values : np.darray
        categorial values of a column into np array format
        example : values = np.array(df[['Type']])
    index_df : pandas indexes
        index of the one hot encoded values
        Must be the same size as the concatenated dataframe
        example : index_df = df.index -> df is the original dataframe
    columns_df : columns of the one-hot-encoded values
        Length of columns must be the same size as the number of unique value in a categorical variable
        exameple : columns_df = np.sort(df['Type'].unique())

    Return
    ------
    Dataframe of the one-hot-encoded column

    Example
    -------
    values = np.array(df[['Type']]) 
    index_df = df.index
    columns_df = np.sort(df['Type'].unique())

    df_ohe = onehot_encoder(values, index_df, columns_df)
    df_final = pd.concat([df_final, df_ohe], axis=1).drop("Type", axis=1)
    """
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    return pd.DataFrame(data=onehot_encoded, index=index_df, columns=columns_df).astype(int)
