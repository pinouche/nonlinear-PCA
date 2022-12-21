import pandas as pd

from datasets.synthetic_datasets import make_two_spheres, make_alternate_stripes, circles_data


def load_data(dataset):

    if dataset == "spheres":
        data = make_two_spheres()

    elif dataset == "circles":
        data = circles_data()

    elif dataset == "alternate_stripes":
        data = make_alternate_stripes()

    elif dataset == "abalone":
        data = pd.read_csv("datasets/abalone.data")

    else:
        raise ValueError(f"dataset {dataset} is not valid.")

    return data