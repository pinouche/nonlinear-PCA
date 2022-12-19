import numpy as np


def split_data(data, val_prop: float = 0.2):

    num_examples = data.shape[0]
    random_index = np.arange(num_examples)
    np.random.shuffle(random_index)
    val_data = data[random_index[:int(num_examples*val_prop)]]
    train_data = data[random_index[int(num_examples*val_prop):]]

    return train_data, val_data
