import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from module_utils.scoreRegression import scoreClassif
from sklearn.model_selection import train_test_split
from sklearn import metrics


def train_model(model, x_train, y_train):
    """Train the model by calling the fit function

    Parameters
    ----------
    model (object) : The object to fit the data.
    x_train : array-like.
        The data to fit. Can be for example a list, or an array.
    y_train : array-like.
        The target varaible to try to predict in the case of supervised learning.

    Returns
    -------
    None"""
    model.fit(x_train, y_train)


def compute_losses_reg(model, x, y):
    """Return a dictionary with the metrics of the model

    Parameters
    ----------
    model (object) : The object to predict the  data. Compute y_pred of the model
    x : array-like.
        x is a list or an array that contain the independent variable. To predict the output variable.
    y : array-like.
        y is a list or an array that contain the dependent variable (output). Used to compute metrics, with y_pred

    Returns
    -------
    resume_metrics_reg : dictionary of performance metrics of regression model
        The metrics values
    """
    y_pred = model.predict(x)
    resume_metrics_reg = {}
    resume_metrics_reg["MAE"] = metrics.mean_absolute_error(y, y_pred)
    resume_metrics_reg["MSE"] = metrics.mean_squared_error(y, y_pred)
    resume_metrics_reg["RMSE"] = np.sqrt(metrics.mean_squared_error(y, y_pred))
    resume_metrics_reg["R2"] = model.score(x, y)
    #resume_metrics_reg['Score Classifier'] = scoreClassif(y, y_pred)
    return resume_metrics_reg


def compute_losses_train_test_reg(model, x_train, y_train, x_test, y_test):
    """Call the function compute_losses_reg.

    Parameters
    ----------
    model (object) : The object to predict the data. Compute y_pred of the model

    After train_test_split the dataset into 4 pieces (x_train, x_test, y_train, y_test). 
    x_train : array-like.
        It represent the independent variables in the training set.
    y_train : array-like.
        It represent the dependent variables in the training set.
    x_test : array-like.
        It represent the independent variables in the testing set.
    y_test : array-like.
        It represent the dependent variables in the testing set.

    Returns
    -------
    df : dataframe with performances metrics of training and testing set for regression model.
    """
    losses_train = compute_losses_reg(model, x_train, y_train)
    losses_test = compute_losses_reg(model, x_test, y_test)

    df = {
        'Train': losses_train,
        'Test': losses_test
    }

    df = pd.DataFrame.from_dict(df)

    return df


def plot_mae(y_true, y_pred, ylabel, title_mae='Mean Error and Standard Deviation of Predicted Data'):
    mae = np.abs(y_true - y_pred).mean(axis=0)
    std = np.abs(y_true - y_pred).std(axis=0)
    x_pos = np.arange(y_true.shape[1])

    fig = plt.figure(figsize=(17, 8))
    plt.suptitle(title_mae, y=0.94, fontsize=16, weight='bold')
    for y in range(y_true.shape[1]):
        ax = fig.add_subplot(1, len(y_true[1]), (y + 1))
        ax.bar(x_pos[y], mae[y], yerr=std[y], align="center",
               alpha=0.8, ecolor="black", capsize=10)
        # ax.set_xlabel(xlabel[y], fontsize=16)
        ax.set_ylabel(ylabel[y], fontsize=16)


def plot_predicted(y_true, y_pred, ylabel, title_pred="Actual Data versus Predicted Data"):
    fig = plt.figure(figsize=(17, 8))
    plt.suptitle(title_pred, fontsize=16, weight="bold", y=0.94)
    for y in range(y_true.shape[1]):
        ax = fig.add_subplot(1, 3, (y + 1))
        sns.stripplot(y=y_true[:, y], label='Observations')
        sns.stripplot(y=y_pred[:, y], color='orange',
                      label='Predicted', alpha=0.7)
        #ax.set_xlabel(xlabel[y], fontsize=16)
        ax.set_ylabel(ylabel[y], fontsize=16)
        plt.legend()


def plot_mae_and_predict(model, x, y, ylabel, title_mae='Mean Error and Standard Deviation of Predicted Data', title_pred="Actual Data versus Predicted Data"):
    """1st plot: Plot the MAE (Mean Average Error) of a learning algorithm, and display the mean and variance errors of the output variable.
    2nd plot: Plot strippplot of y and y_pred for each output variable. Used to compare how well the prediction is.

    Parameters
    ----------
    model (object) : use the predict method of the model object, to predict the data. Compute y_pred of the model.

    x : array-like.
        It represent the independent variables in the dataset.

    y : array-like.
        It represent the dependent variables in the dataset.

    y_label : string or list of string to label the ordinate axis

    title_mae (str) : title for the 1st plot

    title_pred (str) : title for the 2nd plot

    Return
    ------
    - return bar plot of mean and variance error
    - return strip plot of real observation and predicted value
    """
    # compute y_pred to predict the value of output variable
    y_pred = model.predict(x)

    # plot mae metric for each output variable
    plot_mae(y, y_pred, ylabel)

    # plot predicted
    plot_predicted(y, y_pred, ylabel)


def split_train_multiple_models(*models, x, y, col_name):
    """
    Parameters
    ----------
    *models : list of models object
    x : numpy.darray
        input variables
    y : numpy.darray
        target variables
    col_name : list of strings
        name of columns are the name of the models

    Return
    ------
    Return dataframe with input data trained by multiple models (*models)
    """
    # tts dataset
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=42)

    # create dataframe which groups together the metrics of all the prediction models
    final_df = pd.DataFrame()
    for model in models:
        # enter x_train and y_train in each of the models
        train_model(model, x_train, y_train)
        dict_losses = compute_losses_reg(model, x_test, y_test)
        df = pd.DataFrame.from_dict({"df": dict_losses})
        final_df = pd.concat([final_df, df], axis=1)
    final_df.columns = [col for col in col_name]
    return final_df
