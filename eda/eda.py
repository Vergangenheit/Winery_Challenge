import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

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


def analyse_continuous(df: pd.DataFrame, var: str):
    df = df.copy()
    df[var].hist(bins=20)
    plt.ylabel('Number of wines')
    plt.xlabel(var)
    plt.title(var)
    plt.show()


def plot_cont_var(df: pd.DataFrame):
    # list of numerical variables
    num_vars = [var for var in df.columns if df[var].dtypes != 'O']
    for var in num_vars:
        analyse_continuous(df, var)


# let's explore the relationship between the wine price and the transformed numerical variables
# with more detail
def relation_cont_target(num_vars: list, df: pd.DataFrame):
    def transform_analyse_continuous(d: pd.DataFrame, v: str):
        d = d.copy()

        # log does not take negative values, so let's be careful and skip those variables
        if 0 in d[v].unique():
            pass
        else:
            # log transform
            d[v] = np.log(d[v])
            # df['price'] = np.log(df['price'])
            plt.scatter(d[v], d['price'])
            plt.ylabel('price')
            plt.xlabel(v)
            plt.show()

    for var in num_vars:
        if var != 'price':
            transform_analyse_continuous(df, var)


def plot_cont_target(df: pd.DataFrame, target:str):
    num_vars = [var for var in df.columns if df[var].dtypes != 'O']
    for var in num_vars:
        if var != target:
            sns.boxplot(x='points', y=df[target], data=df, palette=sns.color_palette('RdBu', 5))
