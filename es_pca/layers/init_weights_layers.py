from es_pca.layers.layers import ForwardLayer, BatchNormLayer


def create_nn_for_numerical_col(n_features, n_layers, hidden_size, batch_norm, activation="leaky_relu", init_mode="identity"):
    layers_list = list()

    layers_list.append(ForwardLayer(n_features, hidden_size, activation, init_mode=init_mode))
    if batch_norm:
        layers_list.append(BatchNormLayer(hidden_size))

    for _ in range(n_layers - 1):
        layers_list.append(ForwardLayer(hidden_size, hidden_size, activation))
        if batch_norm:
            layers_list.append(BatchNormLayer(hidden_size))

    layers_list.append(ForwardLayer(hidden_size, 1, 'identity', init_mode=init_mode))

    return layers_list
