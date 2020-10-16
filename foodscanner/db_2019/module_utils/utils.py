import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
from IPython.display import display

sns.set(font_scale=1.2, style="darkgrid",
        palette="colorblind", color_codes=True)

"""
%reload_ext autoreload
%autoreload 2
"""


def head_shape(df, n=5):
    return display(df.head(n), df.shape)


def describe_plus(df):
    """Generate descriptive statistics. Return dataframe.describe() transposed with the values rounded for better readability.

    Parameters
    ----------
    df : <dataframe>

    Returns
    -------
    Return dataframe"""
    return df.describe().transpose()[['count', 'mean', 'std', 'min', 'max', '25%', '50%', '75%']].apply(lambda x: round(x, 2))


def unique_counts(tab):
    """Count total for each unique feature in column."""
    return dict(zip(np.unique(tab, return_counts=True)[0], np.unique(tab, return_counts=True)[1]))


def na_per_col(dataframe, ascending=True):
    """Display the percentage of missing values per columns in a dataframe (sort Ascending on True)

    Parameters
    ----------
    df : <dataframe>

    Returns
    -------
    Return columns with the percentage of missing values
    """
    return (dataframe.isna().sum()/dataframe.shape[0]).sort_values(ascending=True)


def total_na_df(dataframe):
    percent_na = round(((dataframe.isna().sum().sum()) /
                        (np.prod(dataframe.shape))) * 100, 2)
    print(
        "The dataframe is made up of {0}% missing values.".format(percent_na))


def r2(x, y):
    """Compute coefficient of determination r2 to see if two variables are correlated.

    Parameters
    ----------
    x : <numpy.darray>
        1st variable
    y : <numpy.darray>
        2nd variable

    Return
    ------
    Return coefficient of determination."""
    return stats.pearsonr(x, y)[0] ** 2


def quantile(y):
    """Classified target variable into binary value (0 or 1), by setting a threshold at the 50th quantile.

    Parameter
    ---------
    y : <float or numpy.ndarray of floats>
        Independent variable to transform into binary value (0 or 1)

    Return
    ------
    Return list of classification
    """
    threshold = np.percentile(y, 50, axis=0)
    y = np.where(y < threshold, 0, 1)
    return y
# ----------------------------------------
# ---------- DF.DESCRIBE_OBJECT ----------
# ----------------------------------------


def describe_object(df):
    """
    return a dataframe with count, unique value and frequency of categorical feature (object)
    """
    return df.astype("object").describe().T

# ----------------------------------
# ---------- VALUE COUNTS ----------
# ----------------------------------


def cat_value_counts(df, cols):
    """
    """
    for col in cols:
        value = df[col].value_counts(normalize=True) * 100
        print("\n" "For"+" "+col)
        print(value)

# -----------------------------------------
# ---------- PLOT MISSING VALUES ----------
# -----------------------------------------
# USE MISSINGNO


def heatmap_isna(df, figsize=(16, 8)):
    """

    Parameters
    ----------

    Return
    ------

    """
    sns.set(font_scale=1)
    cmap = sns.cubehelix_palette(start=2.8, rot=.1, light=0.95, n_colors=2)
    grid_kws = {'width_ratios': (0.9, 0.03), 'wspace': 0.1}
    fig, (ax, cbar_ax) = plt.subplots(
        1, 2, gridspec_kw=grid_kws, figsize=figsize)

    # heatmap on dataframe to spot NaN
    ax = sns.heatmap(df.isna(), ax=ax, yticklabels=False, cbar_ax=cbar_ax, cmap=ListedColormap(cmap),
                     cbar_kws={'orientation': 'vertical'})

    # set legend bar label
    cbar_ax.set_yticklabels(['NOT MISSING', 'MISSING'], fontsize=12)
    cbar_ax.yaxis.set_ticks([0.25, 0.75])

    # set title, x and y labels
    ax.set_title('MISSING VALUES IN THE DATAFRAME',
                 fontsize=16, fontweight="bold")
    ax.set_ylabel('ROWS', fontsize=14)
    ax.set_xlabel('COLUMNS', fontsize=14)
# --------------------------------------------------
# ---------- DATAFRAME FOR MISSING VALUES ----------
# --------------------------------------------------
#


def missing_values_table(df):
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'})
    mis_val_table_ren_columns = mis_val_table_ren_columns[
        mis_val_table_ren_columns.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
    print("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
          "There are " + str(mis_val_table_ren_columns.shape[0]) +
          " columns that have missing values.")
    return mis_val_table_ren_columns
# missing_values_table(df)


# ---------------------------------
# ---------- DISTRBUTION ----------
# ---------------------------------


def plot_dist(df, ylabel="Sample Size", title="Distribution of Target Variable", figsize=(17, 6)):
    """Plots a histogram for each columns of a dataframe

    Parameters
    ----------
    df : <Dataframe>
        The pandas object holding the data
    ylabel : The sample size (ordinate axis)
    title : title of the plot

    Returns
    -------
    return matplotlib histogram
    """
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=16, weight="bold")
    for col in np.arange(df.shape[1]):
        ax = fig.add_subplot(1, df.shape[1], (col + 1))
        ax.hist(df[df.columns[col]])
        ax.set_xlabel(df.columns[col], fontsize=14)
        ax.set_ylabel('Sample Size', fontsize=14)


def dist_quantitative_features(df, quant_feat, title, xlabel, ylabel, figsize=(18, 10)):
    """Plots a histogram for each columns of a dataframe

    Parameters
    ----------
    df : <Dataframe>
        The pandas object holding the data

    Returns
    -------

    """
    fig = plt.figure(figsize=figsize)
    for element in quant_feat:
        sns.distplot(df[element].dropna(), hist=True, kde=False, label=element)
        plt.title(title, fontsize=16, fontweight="bold")
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend()

# ---------------------------------------------------------------------
# ---------- PLOT MEAN AND VARIANCE OF QUANTITATIVE FEATURES ----------
# ---------------------------------------------------------------------


def plot_mean_std(df, xlabel="Wavelength", ylabel='Absorbance', title=None, figsize=(15, 4)):
    """Fill the area between two horizontal curves. It allows us to analyse the mean and standard deviation of a target column.
    The function compute the mean and standard deviation of each column of the dataframe then plot it in a fill_between matplotlib func.

    Parameters
    ----------
    df : <Dataframe>
        The pandas object holding the data
    xlabel : <str>
        A label of x-axis
    ylabel : <str>
        A label of y-axis

    Return
    ------
    Return  plot of mean and variance of dataframe."""
    df_describe = describe_plus(df)
    df_mean_std = abs(df_describe[['mean']] / df_describe['mean'].max())
    df_mean_std['std'] = df_describe[['std']] / df_describe['mean'].max()
    mean = df_mean_std['mean']
    std = df_mean_std['std']

    fig, ax = plt.subplots(1, 1, figsize=figsize)
    plt.subplots_adjust(bottom=0.01)
    plt.fill_between(np.arange(
        df_mean_std.shape[0]), (mean-std), (mean + std), alpha=.4, label='std')
    plt.plot(mean, label='mean')
    plt.xticks(rotation=30)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.title(title, fontsize=16, weight='bold')


def three_plot_mean_std(df1, df2, df3, title1, title2, title3, figsize=(15, 4)):
    plot_mean_std(df1, title=title1, figsize=figsize)
    plot_mean_std(df2, title=title2, figsize=figsize)
    plot_mean_std(df3, title=title3, figsize=figsize)

# Same as above but this plot allow to compare mean and std of 2 features


def compare_plot_mean_std(df1, df2, title, xlabel="Wavelength (nm)", ylabel="Absorbance spectra", label_mean="Mean", label_std="Std", label_mean2="2nd Mean", label_std2="2nd Std", figsize=(15, 4)):
    """

    """
    # compute mean and std of df1
    df_describe_1 = describe_plus(df1)
    df_mean_std_1 = abs(df_describe_1[['mean']] / df_describe_1['mean'].max())
    df_mean_std_1['std'] = df_describe_1[['std']] / df_describe_1['mean'].max()
    mean_1 = df_mean_std_1['mean']
    std_1 = df_mean_std_1['std']

    # compute mean and std of df2
    df_describe_2 = describe_plus(df2)
    df_mean_std_2 = abs(df_describe_2[['mean']] / df_describe_2['mean'].max())
    df_mean_std_2['std'] = df_describe_2[['std']] / df_describe_2['mean'].max()
    mean_2 = df_mean_std_2['mean']
    std_2 = df_mean_std_2['std']

    # plot plot
    fig, ax = plt.subplots(1, figsize=figsize)
    plt.subplots_adjust(bottom=0.01)

    plt.fill_between(np.arange(
        df_mean_std_1.shape[0]), (mean_1-std_1), (mean_1 + std_1), alpha=.4, label=label_std)
    plt.plot(mean_1, label=label_mean)

    plt.fill_between(np.arange(
        df_mean_std_2.shape[0]), (mean_2-std_2), (mean_2 + std_2), alpha=.4, label=label_std2)
    plt.plot(mean_2, label=label_mean2)

    plt.xticks(rotation=30)
    plt.xlabel(xlabel, fontsize=16)
    plt.ylabel(ylabel, fontsize=16)
    plt.legend()
    plt.title(title, fontsize=16, weight='bold')

# --------------------------------------------------------------
# --------------- DETECT OUTLIERS WITH BOXPLOT -----------------
# --(OUTLIERS OF QUANTITATIVE FEATURE PER CATEGORICAL FEATURE --
# --------------------------------------------------------------


def plot_boxplot_outliers(dataframe, col, cat, y, title, xlabel, figsize=(18, 10)):
    """

    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.93)
    for i, element in enumerate(cat):
        ax = fig.add_subplot(2, 4, i+1)
        sns.boxplot(y=y, data=dataframe[dataframe[col] == element])
        ax.set_xlabel(element, fontsize=14)
        ax.set_ylabel("")
    fig.text(0.5, 0.065, xlabel, ha='center', va='center', fontsize=14)
    fig.text(0.08, 0.5, y, ha='center', va='center',
             rotation="vertical", fontsize=14)
    _ = plt.show()

# Allows to see potential outliers, the mean and the variance of quantitative features


def violin_strip_quantitative_features(df, quant_feat, title, ylabel, figsize=(18, 10)):
    """
    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.93)
    for i, element in enumerate(quant_feat):
        ax = fig.add_subplot(2, 5, i+1)
        sns.violinplot(y=element, data=df, palette="Set3")
        sns.stripplot(y=element, data=df, palette="Set1", marker="D",
                      edgecolor="gray", alpha=.3)
        ax.set_xlabel(element, fontsize=14)
        ax.set_ylabel("")
    fig.text(0.08, 0.5, ylabel, ha='center', va='center',
             rotation="vertical", fontsize=14)
    _ = plt.show()


def violin_swarm_quantitative_features(df, quant_feat, title, ylabel, figsize=(18, 10)):
    """
    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.93)
    for i, element in enumerate(quant_feat):
        ax = fig.add_subplot(2, 5, i+1)
        sns.violinplot(y=element, data=df)
        sns.swarmplot(y=element, data=df, palette="Set3",
                      edgecolor="gray", alpha=.9)
        ax.set_xlabel(element, fontsize=14)
        ax.set_ylabel("")
    fig.text(0.08, 0.5, ylabel, ha='center', va='center',
             rotation="vertical", fontsize=14)
    _ = plt.show()


def violin_quantitative_features(df, quant_feat, title, ylabel, figsize=(18, 10)):
    """
    """
    fig = plt.figure(figsize=figsize)
    plt.suptitle(title, fontsize=16, fontweight="bold", y=0.93)
    for i, element in enumerate(quant_feat):
        ax = fig.add_subplot(2, 5, i+1)
        sns.violinplot(y=element, data=df)
        ax.set_xlabel(element, fontsize=14)
        ax.set_ylabel("")
    fig.text(0.08, 0.5, ylabel, ha='center', va='center',
             rotation="vertical", fontsize=14)
    _ = plt.show()
