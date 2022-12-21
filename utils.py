import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder


def split_data(data, val_prop: float = 0.2):

    num_examples = data.shape[0]
    random_index = np.arange(num_examples)
    np.random.shuffle(random_index)
    val_data = data[random_index[:int(num_examples*val_prop)]]
    train_data = data[random_index[int(num_examples*val_prop):]]

    return train_data, val_data


def tranform_data_onehot(data):

    data = pd.DataFrame(data)  # here, we decide to work with data frames, so we convert to df
    object_indices = np.where(data.dtypes == 'object')[0]
    data_to_one_hot = data.iloc[:, object_indices]

    num_cols_per_categories = list(data_to_one_hot.nunique())
    cols_to_remove = data_to_one_hot.columns

    enc = OneHotEncoder(handle_unknown='ignore')
    enc.fit(data_to_one_hot)

    data_to_one_hot = enc.transform(data_to_one_hot).toarray()
    data = data.drop(columns=cols_to_remove, inplace=False)
    new_data = np.concatenate((data_to_one_hot, data), axis=1)

    num_cols_per_categories = num_cols_per_categories + [1] * data.shape[1]

    return new_data, num_cols_per_categories

