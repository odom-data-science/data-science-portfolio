import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import savgol_filter
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures

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

# ----------------------------------------------------------------------------------
# --------------- APPLY PREPROCESSING METHOD AND POLYNOMIAL DEGREE -----------------
# ----------------------------------------------------------------------------------


def pipeline_preprocess_poly(x, preprocess, degree):
    """
    Extract x and y from dataframe, process scatter correction method then polynomial degree on input x

    Parameter
    ---------
    x:
    preprocess:
    degree:

    Return
    ------
    Return x transformed and y

    Example
    -------
    #----extract x and y from dataframe - on x we apply msc method and 2nd degree polynomial
    x, y = pipeline_preprocess_poly(df[df["Type"]=="kale"][surface_target], msc, 2)
    #----separate x and y into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    #----Train models using x_train and y_train
    training_models(x=x_train, y=y_train, models=models_list, col_name=models_name)
    """
    x = preprocess(x)

    # Apply polynomial degree
    poly_features = PolynomialFeatures(degree=degree)
    # transforms training and test set to higher degree polynomial
    x = poly_features.fit_transform(x)

    return x


def extract_pipeline_preprocess_poly(df, target, preprocess, degree=1):
    """
    Extract x and y from dataframe, process scatter correction method then polynomial degree on input x

    Parameter
    ---------
    df:
    target:
    col_index:
    preprocess:
    degree:

    Return
    ------
    Return x transformed and y

    Example
    -------
    #----extract x and y from dataframe - on x we apply msc method and 2nd degree polynomial
    x, y = extract_pipeline_preprocess_poly(df=df[df["Type"]=="kale"][surface_target],
                     target=target_variables, col_index=3,
                     preprocess=msc, degree=2)
    #----separate x and y into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
    #----Train models using x_train and y_train
    training_models(x=x_train, y=y_train, models=models_list, col_name=models_name)
    """
    df_copy = df.copy(deep=True)
    df_copy.dropna(inplace=True)
    x = df_copy.drop(target, axis=1).values
    y = df_copy[target].values

    # preprocessing
    x = preprocess(x)

    # Apply polynomial degree
    poly_features = PolynomialFeatures(degree=degree)
    # transforms training and test set to higher degree polynomial
    x = poly_features.fit_transform(x)

    return x, y


def extract_x_y(df, target, col_index):
    """
    Extract x and y from given dataframe
    """
    df.dropna(inplace=True)
    x = df.drop(target, axis=1).values
    y = df[target].iloc[:, col_index].values
    return x, y
