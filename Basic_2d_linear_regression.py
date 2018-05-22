"""
 Basic linear regression for 2 dimensional data + R-square
"""
import sympy as sy
import numpy as np
import pandas as pd


def linear_regression(x, y):
    """
    General linear regression function
    :param x: list of values for the independent variable
    :param y: list of values for the dependent variable
    :return: the slope, intercept and r-square of the linear regression
    """
    df = pd.DataFrame(np.column_stack([x, y])).astype(dtype=float)
    squares_list = squares(df)
    sum_of_squares = sum_squares(squares_list)
    solution = differential_equations(sum_of_squares)
    slope = solution[0]
    intercept = solution[1]
    fitted = fitted_values(slope, intercept, x)
    regression_ss = regression_sum_square(fitted)
    total_ss = total_sum_square(y)
    r_square = regression_ss / total_ss
    slope = format(slope, '.4f')
    intercept = format(intercept, '.4f')
    r_square = format(r_square, '.4f')
    print("The slope is {}, the intercept is {}, and the R-square is {}".format(slope, intercept, r_square))


def squaring(row):
    """
    Gets the square of the difference between y and ^y for each row
    :param row: a row in the data
    :return: (a * Y0 + b - X0)^2 as a string
    """
    sq = ''.join(['(a * ', str(row[0]), ' + b - ', str(row[1]), ') ^ 2'])
    return sq


def squares(df):
    """
    :type df: pd.dataframe
    :param df: data frame of data
    :return: the data frame with an extra column with the difference between y and ^y
    """
    squares_list = df.apply(squaring, axis=1)
    return squares_list


def sum_squares(df):
    """
    :param df: data frame with the appended column of equations
    :return: a string adding all the equations
    """
    sum_of_squares = []
    for square_row in df:
        result = ''.join([square_row, ' + '])
        sum_of_squares.append(result)
    sum_of_squares = ''.join(sum_of_squares)
    sum_of_squares = sum_of_squares[:-3]
    return sum_of_squares


def differential_equations(sum_of_squares):
    """
    Sets up and solves the differential equations
    :param sum_of_squares: the sum of squares as a function of a and b
    :return: the solutions for the partial derivatives equations
    """
    a, b = sy.symbols('a b')
    differential_a = sy.diff(sum_of_squares, a)
    differential_b = sy.diff(sum_of_squares, b)
    solution = list(sy.linsolve([differential_a, differential_b], (a, b)))
    return solution[0]


def fitted_values(slope, intercept, x):
    fitted_val = [value * slope + intercept for value in x]
    return fitted_val


def regression_sum_square(fitted):
    squared_errors_reg = [x - np.mean(fitted) for x in fitted]
    regression_SS = sum(squared_errors_reg)
    return regression_SS


def total_sum_square(y):
    squared_errors_tot = [x - np.mean(y) for x in y]
    total_SS = sum(squared_errors_tot)
    return total_SS


"""
Test
"""
X0 = [1, 2, 3, 4, 5, 6, 7, 8, 9]
Y0 = [2, 3, 4, 6, 4, 6, 8, 5, 8]

linear_regression(X0, Y0)  