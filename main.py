from eda import eda
from feature_engineering import feature_eng
from model import build_model
import pandas as pd
import numpy as np

data = pd.read_csv('data/assignment_dataset.csv')

discrim = np.isnan(data.price.values)
train = data[~discrim].reset_index(drop=True)
test = data[discrim].reset_index(drop=True)

# PLOT PRICE
eda.plot_price(train)
# ANALYZE AND REMOVE OUTLIERS
eda.show_price_outliers(train)
eda.remove_price_outliers(train)

# ANALYZE NAN VALUES
eda.plot_var_with_na(train)

# ANALYZE CONTINUOUS VAR
eda.plot_cont_var(train)

# list of numerical variables
num_vars = [var for var in train.columns if train[var].dtypes != 'O']
eda.relation_cont_target(num_vars, train)

eda.plot_cont_target(train, 'price')