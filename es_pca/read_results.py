import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os

from es_pca.utils import config_load

CONFIG = config_load()


def parse_arguments():
    parser = argparse.ArgumentParser(description='give arguments to display results')
    parser.add_argument('--results_file_path', nargs='+', type=str, help='file path')
    parser.add_argument('--dataset', type=str, help='dataset name')

    arguments = parser.parse_args()

    return arguments


def load_data(path: str) -> list:
    data_results = pickle.load(open(path, "rb"))

    return data_results


def compute_quantiles(fitness_list: np.array) -> tuple[np.array, np.array, np.array]:

    percentiles = (np.quantile(fitness_list, axis=0, q=0.2),
                   np.quantile(fitness_list, axis=0, q=0.5),
                   np.quantile(fitness_list, axis=0, q=0.8))

    return percentiles


def retrieve_data(results_dictionary: dict, key: str) -> tuple[np.array, np.array]:

    objective_list = [results_dictionary[key][index][0] for index in range(CONFIG["number_of_runs"])]
    objective_val = np.reshape(np.array([tup[1] for element in objective_list for tup in element]),
                               (CONFIG["number_of_runs"], CONFIG["epochs"]))
    objective_train = np.reshape(np.array([tup[0] for element in objective_list for tup in element]),
                                 (CONFIG["number_of_runs"], CONFIG["epochs"]))

    return objective_val, objective_train


def plot_quantiles(results_dictionary: dict, dataset_name: str = None) -> None:

    # CB_color_cycle = ['#377eb8', '#ff7f00', '#4daf4a',
    #                   '#f781bf', '#a65628', '#984ea3',
    #                   '#999999', '#e41a1c', '#dede00']

    color_list = ["darkblue", "darkgreen", "darkred", "darkorange"]
    legend_list = ["h=cos, objective=full",
                   "h=cos, objective=partial",
                   "h=ReLU, objective=full",
                   "h=ReLU, objective=partial"]
    legend_entries = []
    plt.figure(figsize=(10, 10))
    plt.ylim(1, 2.1)
    plt.xlim(1, CONFIG["epochs"])

    enumerate_counter = 0
    for key, value in results_dictionary.items():
        objective_val, objective_train = retrieve_data(results_dictionary, key)

        percentiles_val = compute_quantiles(objective_val)
        percentiles_train = compute_quantiles(objective_train)

        color = color_list[enumerate_counter]

        plt.plot(np.arange(1, CONFIG["epochs"] + 1, 1), percentiles_val[1], color=color)
        plt.plot(np.arange(1, CONFIG["epochs"] + 1, 1), percentiles_train[1], color=color, linestyle='--')
        legend_entries.append(legend_list[enumerate_counter])
        legend_entries.append(f"_")
        # plt.fill_between(np.arange(1, CONFIG["epochs"] + 1, 1), percentiles_val[0] + 0.001, percentiles_val[2] - 0.001,
        #                  color=color, alpha=0.2)
        # plt.fill_between(np.arange(1, CONFIG["epochs"] + 1, 1), percentiles_train[0] + 0.001,
        #                  percentiles_train[2] - 0.001, color=color, alpha=0.2)
        # legend_entries.append(f"_")
        # legend_entries.append(f"_")

        enumerate_counter += 1

    plt.legend(legend_entries, loc="upper left", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(np.arange(0, CONFIG["epochs"] + 1, 20), fontsize=16)
    plt.xlabel('Generations', size=16)
    plt.ylabel("$\mathcal{F}_{\mathrm{total}}^{1}$", size=17)
    plt.grid(True)

    path_to_save = f"./results/plots/{dataset_name}/quantiles_plot.pdf"
    directory = os.path.dirname(path_to_save)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(path_to_save, dpi=300)  # Adjust dpi if needed


def plot_2d_scatter(results_list: list, dataset: str, size: int) -> None:

    val_data_pca = [[] for _ in range(len(results_list))]
    train_data_pca = [[] for _ in range(len(results_list))]

    for run_number in range(len(results_list)):
        for gen_number in range(len(results_list[0])):
            val_data_pca[run_number].append(results_list[run_number][gen_number].objectives[1][2])
            train_data_pca[run_number].append(results_list[run_number][gen_number].objectives[0][2])

    for run_number in range(len(results_list)):
        for gen_n in [0, 25, 50, 75, len(results_list[0])-1]:

            plt.figure(figsize=(20, 10))
            plt.grid(True)
            plt.scatter(val_data_pca[run_number][gen_n][:, 0], val_data_pca[run_number][gen_n][:, 1], c="red", s=20, edgecolor="k", label="Validation data")
            plt.scatter(train_data_pca[run_number][gen_n][:, 0], train_data_pca[run_number][gen_n][:, 1], c="blue", s=20, edgecolor="k", alpha=0.8, label="Training data")
            plt.xticks(size=14)
            plt.yticks(size=14)
            plt.xlabel("$x_1$", size=15)
            plt.ylabel("$x_2$", size=15)
            plt.legend(fontsize=15)

            path_to_save = f"./results/plots/scatter_transformed/pca_{dataset}/max_tree_size_{size}/run_number_{run_number}/gen_n_{gen_n}.pdf"
            directory = os.path.dirname(path_to_save)
            if not os.path.exists(directory):
                os.makedirs(directory)

            plt.savefig(path_to_save, dpi=300)
            plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = args.dataset

    results_dic = {}
    base_path = f"./results/{dataset}/"
    for activation in ["cos", "relu"]:
        for partial_contrib in ["False", "True"]:
            list_runs = []
            file_path = base_path + f"activation={activation}/partial_contrib={partial_contrib}"
            for run_number in range(CONFIG["number_of_runs"]):
                data = pickle.load(open(file_path+f"/{run_number}.p", "rb"))
                list_runs.append(data)
            results_dic[file_path] = list_runs

    #  objective_list, (x_transformed_train, x_transformed_val, pca_transformed_train, pca_transformed_val)

    print(results_dic.keys())
    plot_quantiles(results_dic, dataset)
