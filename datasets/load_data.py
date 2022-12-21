


def load_data(dataset):

    if dataset == "spheres":
        data_x = make_two_spheres()

    elif dataset == "circles":
        data_x = circles_data()

    elif dataset == "alternate_stripes":
        data_x = make_alternate_stripes()

    else:
        raise ValueError(f"dataset {dataset} is not valid.")

    return data_x