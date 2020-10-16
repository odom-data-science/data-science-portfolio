import numpy as np
import pandas as pd


def scoreClassif(y_true, y_pred):
    """On entre en paramètre y_true et y_pred qui ont des valeurs continues, ensuite
    on détermine le threshold pour classifier les variables en O ou 1"""
    threshold = np.percentile(y_true, 50, axis=0)
    y_true_classif = y_true < threshold
    y_pred_classif = y_pred < threshold
    return np.sum(y_pred_classif == y_true_classif) / np.prod(y_pred.shape)


def scoreClassifier(model, x, y):
    """En paramètre y_true et y_pred qui ont des valeurs binaire (0 ou 1), pour
    déterminer le proportion de bonne prediction,"""
    # On reshape y_pred pour le mettre dans la même shape que y_true pour pouvoir les comparer
    y_pred = model.predict(x)
    y_pred = y_pred.reshape(y.shape[0], y.shape[1])
    return np.sum(y_pred == y) / np.prod(y_pred.shape)

# Pre-processing


def one_hot_encoder(df, column_list):
    """Takes in a dataframe and a list of columns
    for pre-processing via one hot encoding"""
    df_to_encode = df[column_list]
    df = pd.get_dummies(df_to_encode)
