"""
Series of function that calculates the gradient descent for a simple linear regression
Can be used for vanilla, stochastic, and mini-batch gradient descent
There is no correction of the learning rate, but it could be added in the function update_coefficients
It is applied to the white wine data set from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
"""

import pandas as pd
import numpy as np


def gradient_descent(data, learning_rate, coefficients, batch_size):
    """
    Function that actually does the gradient descent
    :param data: data frame of the data on which to do the gradient descent
    :param learning_rate: best set at 0.0001 with the wine data and no correction
    :param coefficients: starting coefficients
    :param batch_size: Make it a vanilla gradient descent by setting it to len(data)
            Set to 1 for a stochastic gradient descent, and anything in between for a mini batch
    :return: returns the coefficients and the list of mean squared errors (one mse per batch)
    """
    data, n_features = preparing_df(data)
    data = fitted_values(data, coefficients)
    mean_squared_errors = []
    # Creating lists of row indexes for the batches
    index_batches = []
    indexes_rows = list(range(len(data) - 1))
    for i in range(0, len(indexes_rows), int(len(data) / (batch_size - 1))):
        batch = indexes_rows[i:i + int(len(data) / (batch_size - 1))]
        index_batches.append(batch)
    for index_batch in index_batches:
        batch = data.loc[index_batch, :]
        batch = fitted_values(batch, coefficients)
        batch = gradient(batch, n_features)
        coefficients = update_coefficients(batch, coefficients, learning_rate)
        mse = np.mean((batch['fitted'] - batch['observed']) ** 2)
        mean_squared_errors.append(mse)
    return coefficients, mean_squared_errors


def repeated_gradient_descent(data, learning_rate, batch_size, n_epochs=1):
    """
    Function to call to repeat the gradient descent for several epochs
    :param n_epochs: number of epochs (of times) for the gradient descent
    FOR THE OTHER PARAMETERS: see gradient_descent(data, learning_rate, coefficients, batch_size=len(data))
    :return: the final coefficients and the list of all mean squared error (for each batch in each epoch)
    """
    n_features = len(data.columns[:-1])
    coefficients = [0.0] * (n_features + 1)
    list_mse = []
    for n in range(n_epochs):
        coefficients, mean_squared_errors = gradient_descent(data, learning_rate, coefficients, batch_size)
        list_mse.append(mean_squared_errors)
    return coefficients, list_mse


def preparing_df(data):
    """
    Rearranges the data to make it easier to call the columns later
    :param data: Dataframe with the response variable in the last column
    :return: the reformated data frame, and the number of features
    """
    # Getting the number of features before adding new columns
    n_features = len(data.columns[:-1])
    # Renaming the columns of features to make it easier to call them later
    colnames = []
    n = 0
    for column in data.columns[:-1]:
        n += 1
        column = 'feature_' + str(n)
        colnames.append(column)
    colnames.append('observed')
    data.columns = colnames
    # Creating columns for the gradient
    n = 0
    for x in range(0, n_features + 1):
        name_column = 'gradient_feature_' + str(n)
        data[name_column] = 0
        n += 1
    return data, n_features


def fitted_values(data, coefficients):
    """
    Calculates the fitted values for the data based on the coefficients
    :param data: the data frame (or partial dataframe) for which to calculate the fitted values
    :param coefficients: the current value for the coefficients
    :return: the data frame with an added (or updated) column containing the fitted values
    """
    #intercept
    data['fitted'] = coefficients[0]
    #rest
    for n in range(1, len(coefficients) - 1):
        feature = 'feature_' + str(n)
        data['fitted'] += coefficients[n + 1] * data[feature]
    return data


def gradient(data, n_features):
    """
    Calculates the gradient by calculating, for each Xi and for each feature, the Xi's contribution
    to the gradient (that can then just be summed later to get the actual gradient)
    :param data: dataframe or partial data frame with the data and the fitted values
    :param n_features: number of features (as it cannot be extrapolated from the number of columns anymore)
    :return: the dataframe with the gradient contributions updated/added
    """
    colnames = []
    n = 0
    for x in range(0, n_features + 1):
        column = 'gradient_feature_' + str(n)
        colnames.append(column)
        n += 1
    data[colnames[0]] = data['fitted'] - data['observed']
    for column in colnames[1:]:
        # The last term cuts the 'gradient_' out of the column name to get the column with the feature
        data[column] = (data['fitted'] - data['observed']) * data[column[9:]]
    return data


def update_coefficients(data, coefficients, learning_rate):
    """
    Updates the coefficients, and a correction could be added here
    :param data: data frame with the columns for the gradient contributions of each Xi
    :param coefficients: coefficients at time t
    :param learning_rate: best set at 0.0001 with the wine data and no correction
    :return: the list of coefficients at time t + 1
    """
    for n in range(len(coefficients)):
        gradient_col = 'gradient_feature_' + str(n)
        coefficients[n] += learning_rate * sum(data[gradient_col])
    return coefficients


data = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv')
results = repeated_gradient_descent(data, 0.000001, batch_size=50, n_epochs=2)
print(results)



