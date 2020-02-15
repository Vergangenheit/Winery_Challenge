import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt

"""plot price"""


def plot_price(df: pd.DataFrame):
    plt.subplot(1, 2, 1)
    df['price'].hist(bins=50, figsize=(12, 6))
    plt.title('Price Distribution', fontsize=12)

    plt.subplot(1, 2, 2)
    np.log(df['price']).hist(bins=50, figsize=(12, 6))
    plt.title('Log-Price Distribution', fontsize=12)


def show_price_outliers(df: pd.DataFrame):
    df = df.copy()
    df['price'] = np.log(df['price'])
    df.boxplot(column='price')
    plt.title('Price')
    plt.ylabel('price')
    plt.show()


def remove_price_outliers(df: pd.DataFrame):
    price_mean = np.mean(df.price.values)
    std_price = np.std(df.price.values)
    price_outliers = np.logical_and(df.price.values < price_mean + 3 * std_price,
                                    df.price.values > price_mean - 3 * std_price)
    df = df.copy()
    df = df[price_outliers]


def analyse_na_value(df: pd.DataFrame, var: str):
    df = df.copy()

    # let's make a variable that indicates 1 if the observation was missing or zero otherwise
    df[var] = np.where(df[var].isnull(), 1, 0)

    # let's calculate the mean SalePrice where the information is missing or present
    df.groupby(var)['price'].median().plot.bar()
    plt.title(var)
    plt.show()


def plot_var_with_na(df: pd.DataFrame):  # make a list of the categorical variables that contain missing values
    vars_with_na = [var for var in df.columns if df[var].isnull().sum() > 1]
    for var in vars_with_na:
        analyse_na_value(df, var)
