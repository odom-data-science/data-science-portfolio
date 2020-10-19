import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split

from module_utils.model_training import train_model, compute_losses_reg
from module_utils.utils import plot_mean_std


def array_to_df(input_data, preprocess, index, columns):
    """
    Apply preprocess method on input_data and use output to create a new dataframe

    Parameters
    ----------
    input_data : <numpy.darray>
    preprocess : <function>
        msc(), snv()
    index : <indexes.numeric>
        used as index for new dataframe with transformed input data
    columns : <columns name>
        list of strings

    Returns
    -------
    transformed_data : input data after preprocess method
    transformed_df : dataframe with new value of input data
    """
    transformed_data = preprocess(input_data)
    transformed_df = pd.DataFrame(
        data=transformed_data, index=index, columns=columns)
    return transformed_data, transformed_df


def msc(input_data, reference=None):
    """Perform Multiplicative scatter correction
    Parameters
    ----------
    input_data : <numpy.darray>
        NIRS data matrix

    Return
    ------
    Return 
    """

    # mean centre correction
    for i in range(input_data.shape[0]):
        input_data[i, :] -= input_data[i, :].mean()

    # Get the reference spectrum. If not given, estimate it from the mean
    if reference is None:
        # calculate mean
        ref = np.mean(input_data, axis=0)
    else:
        ref = reference

    # Define a new array and populate it with the corrected data
    data_msc = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Run regression
        fit = np.polyfit(ref, input_data[i, :], 1, full=True)
        # Apply correction
        data_msc[i, :] = (input_data[i, :] - fit[0][1]) / fit[0][0]

    return data_msc


def snv(input_data):
    """Apply Standard Normal Variate

    Parameters
    ----------
    input_data : <numpy.darray>
        NIRS data matrix

    Return
    ------
    Return 
    """

    # Define a new array and populate it with the corrected data
    output_data = np.zeros_like(input_data)
    for i in range(input_data.shape[0]):
        # Apply correction
        output_data[i, :] = (
            input_data[i, :] - np.mean(input_data[i, :])) / np.std(input_data[i, :])

    return output_data

#
# def savgol_func(X, window_length=11, polyorder=3, deriv_order=0, delta=1.0):
#    return X.apply(lambda x: savgol_filter(x, window_length, polyorder))
#


def savgol_smoothing(x, window_length=5, polyorder=3, deriv_order=0, delta=1.0):
    return savgol_filter(x, window_length, polyorder)


def detrend(spectra, bp=0):
    """ Perform spectral detrending to remove linear trend from data.
    Parameters
    ----------
    spectra : <numpy.ndarray>
        NIRS data matrix
    bp : <list>
        A sequence of break points. If given, an individual linear fit is performed for each part of data
        between two break points. Break points are specified as indices into data.
    Return
    ------
    Return spectra <numpy.ndarray>: Detrended NIR spectra
    """
    return scipy.signal.detrend(spectra, bp=bp)


def norml(spectra, udefined=True, imin=0, imax=1):
    """ Perform spectral normalisation with user-defined limits.
    Parameters
    ----------
        spectra <numpy.ndarray>: NIRS data matrix.
        udefined <bool>: use user defined limits
        imin <float>: user defined minimum
        imax <float>: user defined maximum

    Returns
    -------
        spectra <numpy.ndarray>: Normalized NIR spectra
    """
    if udefined:
        f = (imax - imin)/(np.max(spectra) - np.min(spectra))
        n = spectra.shape
        # create empty array for spectra
        arr = np.empty((0, n[0]), dtype=float)
        for i in range(0, n[1]):
            d = spectra[:, i]
            dnorm = imin + f*d
            arr = np.append(arr, [dnorm], axis=0)
        return np.transpose(arr)
    else:
        return spectra / np.linalg.norm(spectra, axis=0)


def lsnv(spectra, num_windows=10):
    """ Perform local scatter correction using the standard normal variate.
    Parameters
    ----------
        spectra <numpy.ndarray>: NIRS data matrix.
        num_windows <int>: number of equispaced windows to use (window size (in points) is length / num_windows)

    Returns
    -------
        spectra <numpy.ndarray>: NIRS data with local SNV applied.
    """

    parts = np.array_split(spectra, num_windows, axis=0)
    for idx, part in enumerate(parts):
        parts[idx] = snv(part)

    return np.concatenate(parts, axis=0)


def process_n_train(*args, df, y, preprocess, models_name):
    """
    Apply smoothing method on df.values, takes transformed data as input to models

    Parameters
    ----------
    *args : list of objects
    df :
    y : numpy.darray
        target variable - variable to predict
    preprocess : object
        object used to transform data
    models_name : list of strings
        columns of the new dataframe - list of models name

    Returns
    -------
    Dataframe of metrics, plot mean std of transformed data
    """
    index = df.index
    columns = df.columns
    x = df.values

    # apply transformation on x values
    transformed_data = preprocess(x)
    transformed_df = pd.DataFrame(
        data=transformed_data, index=index, columns=columns)

    # plot mean and standard deviation of transformed input data (after preprocessing)
    plot_mean_std(transformed_df)

    # train test split data and enter in list of models (object) *args
    x_train, x_test, y_train, y_test = train_test_split(
        transformed_data, y, test_size=0.30, random_state=42)

    # create a dataframe which groups together the metrics of all the prediction models
    final_df = pd.DataFrame()
    for model in args:
        # enter x_train and y_train in each of the models
        train_model(model, x_train, y_train)
        dict_losses = compute_losses_reg(model, x_test, y_test)
        df = pd.DataFrame.from_dict({"df": dict_losses})
        final_df = pd.concat([final_df, df], axis=1)
    final_df.columns = [col for col in models_name]
    return final_df
