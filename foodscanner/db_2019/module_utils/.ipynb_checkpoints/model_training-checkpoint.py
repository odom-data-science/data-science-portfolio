import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from module_utils.scoreRegression import scoreClassif
from sklearn import metrics


def train_model(model, x_train, y_train):
    model.fit(x_train, y_train)


def compute_losses_reg(model, x, y):
    y_pred = model.predict(x)
    resume_metrics_reg = {}
    resume_metrics_reg["MAE"] = metrics.mean_absolute_error(y, y_pred)
    resume_metrics_reg["MSE"] = metrics.mean_squared_error(y, y_pred)
    resume_metrics_reg["RMSE"] = np.sqrt(metrics.mean_squared_error(y, y_pred))
    resume_metrics_reg["R2"] = model.score(x, y)
    resume_metrics_reg['Score Classifier'] = scoreClassif(
        y, y_pred)
    return resume_metrics_reg


def compute_losses_train_test_reg(model, x_train, y_train, x_test, y_test):
    """
    L'objectif: Fonction qui calcule les metrics pour le train et le test,
    ensuite retourne un dictionnaire qui contient les metrics du train et du test
    """
    losses_train = compute_losses_reg(model, x_train, y_train)
    losses_test = compute_losses_reg(model, x_test, y_test)

    df = {
        'Model Train': losses_train,
        'Model Test': losses_test
    }
    return df


def plot_mae(y_true, y_pred, ylabel, title='Mean Error and Standard Deviation of Predicted Data'):
    mae = np.abs(y_true - y_pred).mean(axis=0)
    std = np.abs(y_true - y_pred).std(axis=0)
    x_pos = np.arange(y_true.shape[1])

    fig = plt.figure(figsize=(17, 8))
    plt.suptitle(title, y=0.94, fontsize=16, weight='bold')
    for y in range(y_true.shape[1]):
        ax = fig.add_subplot(1, len(y_true[1]), (y + 1))
        ax.bar(x_pos[y], mae[y], yerr=std[y], align="center",
               alpha=0.8, ecolor="black", capsize=10)
        # ax.set_xlabel(xlabel[y], fontsize=16)
        ax.set_ylabel(ylabel[y], fontsize=16)


def plot_predicted(y_true, y_pred, ylabel, title="Actual Data versus Predicted Data"):
    fig = plt.figure(figsize=(17, 8))
    plt.suptitle(title, fontsize=16, weight="bold", y=0.94)
    for y in range(y_true.shape[1]):
        ax = fig.add_subplot(1, 3, (y + 1))
        sns.stripplot(y=y_true[:, y], label='Observations')
        sns.stripplot(y=y_pred[:, y], color='orange',
                      label='Predicted', alpha=0.7)
        #ax.set_xlabel(xlabel[y], fontsize=16)
        ax.set_ylabel(ylabel[y], fontsize=16)
        plt.legend()


resume_metrics_reg = {}


def pred_compute_plot_metrics_reg(model, x_test, y_test, ylabel, title_mae='Mean Error and Standard Deviation of Predicted Data', title_pred="Actual Data versus Predicted Data"):
    # predict the value of output variable
    global y_pred  # y_pred is now global
    y_pred = model.predict(x_test)

    # created a dictionary where is stored the metric of our regression algorithm
    global resume_metrics_reg
    resume_metrics_reg = {}
    resume_metrics_reg["MAE"] = metrics.mean_absolute_error(y_test, y_pred)
    resume_metrics_reg["MSE"] = metrics.mean_squared_error(y_test, y_pred)
    resume_metrics_reg["RMSE"] = np.sqrt(
        metrics.mean_squared_error(y_test, y_pred))
    resume_metrics_reg["R2"] = model.score(x_test, y_test)
    resume_metrics_reg['Score Classifier'] = scoreClassif(y_test, y_pred)

    # plot mae metric for each output variable
    mae = np.abs(y_test - y_pred).mean(axis=0)
    std = np.abs(y_test - y_pred).std(axis=0)
    x_pos = np.arange(y_test.shape[1])
    fig = plt.figure(figsize=(17, 8))
    plt.suptitle(title_mae, y=0.94, fontsize=16, weight='bold')
    for y in range(y_test.shape[1]):
        ax = fig.add_subplot(1, len(y_test[1]), (y + 1))
        ax.bar(x_pos[y], mae[y], yerr=std[y], align="center",
               alpha=0.8, ecolor="black", capsize=10)
        # ax.set_xlabel(xlabel[y], fontsize=16)
        ax.set_ylabel(ylabel[y], fontsize=16)
    plt.show()

    fig = plt.figure(figsize=(16, 8))
    plt.suptitle(title_pred, fontsize=16, weight="bold", y=0.94)
    for y in range(y_test.shape[1]):
        ax = fig.add_subplot(1, y_test.shape[1], (y + 1))
        sns.stripplot(y=y_test[:, y], label='Observations')
        sns.stripplot(y=y_pred[:, y], color='orange',
                      label='Predicted', alpha=0.7)
        #ax.set_xlabel(xlabel[y], fontsize=16)
        ax.set_ylabel(ylabel[y], fontsize=16)
        plt.legend()
    plt.show()


def pred_compute_traintest_plot_metrics_reg(model, x_train, y_train, x_test, y_test, ylabel, title_mae='Mean Error and Standard Deviation of Predicted Data', title_pred="Actual Data versus Predicted Data"):
    # predict the value of output variable
    global y_pred  # y_pred is now global
    y_pred = model.predict(x_test)

    # created a dictionary where is stored the metric of our regression algorithm
    global resume_metrics_reg
    resume_metrics_reg = {}
    resume_metrics_reg["MAE"] = metrics.mean_absolute_error(y_test, y_pred)
    resume_metrics_reg["MSE"] = metrics.mean_squared_error(y_test, y_pred)
    resume_metrics_reg["RMSE"] = np.sqrt(
        metrics.mean_squared_error(y_test, y_pred))
    resume_metrics_reg["R2 (train)"] = model.score(x_train, y_train)
    resume_metrics_reg["R2 (test)"] = model.score(x_test, y_test)
    resume_metrics_reg['Score Classifier'] = scoreClassif(y_test, y_pred)

    # plot mae metric for each output variable
    mae = np.abs(y_test - y_pred).mean(axis=0)
    std = np.abs(y_test - y_pred).std(axis=0)
    x_pos = np.arange(y_test.shape[1])
    fig = plt.figure(figsize=(17, 8))
    plt.suptitle(title_mae, y=0.94, fontsize=16, weight='bold')
    for y in range(y_test.shape[1]):
        ax = fig.add_subplot(1, len(y_test[1]), (y + 1))
        ax.bar(x_pos[y], mae[y], yerr=std[y], align="center",
               alpha=0.8, ecolor="black", capsize=10)
        # ax.set_xlabel(xlabel[y], fontsize=16)
        ax.set_ylabel(ylabel[y], fontsize=16)
    plt.show()

    fig = plt.figure(figsize=(16, 8))
    plt.suptitle(title_pred, fontsize=16, weight="bold", y=0.94)
    for y in range(y_test.shape[1]):
        ax = fig.add_subplot(1, y_test.shape[1], (y + 1))
        sns.stripplot(y=y_test[:, y], label='Observations')
        sns.stripplot(y=y_pred[:, y], color='orange',
                      label='Predicted', alpha=0.7)
        #ax.set_xlabel(xlabel[y], fontsize=16)
        ax.set_ylabel(ylabel[y], fontsize=16)
        plt.legend()
    plt.show()
