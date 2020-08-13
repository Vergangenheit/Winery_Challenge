from eda import eda
from feature_engineering.preprocessing import tokenize_feature, split_train_test, fill_categorical_na
from feature_engineering import feature_eng as fe
from model import build_model
import pandas as pd
import numpy as np
import tensorflow as tf

if __name__ == "__main__":
    data = pd.read_csv('data/assignment_dataset.csv')

    discrim = np.isnan(data.price.values)
    train = data[~discrim].reset_index(drop=True)
    test = data[discrim].reset_index(drop=True)

    # PLOT PRICE
    eda.plot_price(train)
    # ANALYZE AND REMOVE OUTLIERS
    eda.show_price_outliers(train)
    train = eda.remove_price_outliers(train)

    # ANALYZE NAN VALUES
    eda.plot_var_with_na(train)

    # ANALYZE CONTINUOUS VAR
    eda.plot_cont_var(train)

    # list of numerical variables
    num_vars = [var for var in train.columns if train[var].dtypes != 'O']
    eda.relation_cont_target(num_vars, train)

    eda.plot_cont_target(train, 'price')

    # PREPROCESS THE DATA

    # make a list of the categorical variables that contain missing values
    vars_with_na = [var for var in train.columns if train[var].isnull().sum() > 1 and train[var].dtypes == 'O']
    # replace missing values with new label: "Missing"
    train = fill_categorical_na(train, vars_with_na)
    test2 = fill_categorical_na(test, vars_with_na)

    # Feature scaling
    train = fe.feature_scaling(train)
    test2 = fe.feature_scaling(test2)

    #Transform categorical variables
    cat_vars = [var for var in train.columns if train[var].dtypes == 'O' and var not in ['title', 'description']]
    train = fe.replace_categoricals(train, cat_vars)
    test2 = fe.replace_categoricals(test2, cat_vars)

    title_pad_train = tokenize_feature(train, 'title')
    title_pad_test = tokenize_feature(test2, 'title')
    desc_train = tokenize_feature(train, 'description')
    desc_test = tokenize_feature(test2, 'description')

    # Stack to build the train and test arrays
    train_features = train.drop(['title', 'description', 'price'], axis=1).values
    test_features = test2.drop(['title', 'description', 'price'], axis=1).values
    features_train = np.hstack((train_features, title_pad_train, desc_train))
    features_test = np.hstack((test_features, title_pad_test, desc_test))
    target = train['price'].values

    #split trainset into train and validation
    X_train, X_valid, y_train, y_valid = split_train_test(features=features_train, target=target)

    # INSTANTIATE MODEL, COMPILE AND TRAIN
    model = build_model(dim=X_train.shape[1])
    model.compile(optimizer='adam', loss=tf.keras.losses.MeanAbsolutePercentageError(),
                  metrics=[tf.keras.metrics.MeanAbsolutePercentageError()])
    model.fit(X_train, y_train,
              validation_data=(X_valid, y_valid),
              batch_size=128,
              epochs=30, verbose=2)
    model.save('./model.hdf5')

    #PREDICT PRICE





