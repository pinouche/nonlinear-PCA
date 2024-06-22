import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt
import os
from loguru import logger

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


def retrieve_objective_data(results_dictionary: dict, key: str) -> tuple[np.array, np.array]:

    objective_list = [results_dictionary[key][index][0] for index in range(CONFIG["number_of_runs"])]
    objective_val = np.reshape(np.array([tup[1] for element in objective_list for tup in element]),
                               (CONFIG["number_of_runs"], CONFIG["epochs"]))
    objective_train = np.reshape(np.array([tup[0] for element in objective_list for tup in element]),
                                 (CONFIG["number_of_runs"], CONFIG["epochs"]))

    return objective_val, objective_train


def plot_quantiles(results_dictionary: dict, dataset_name: str = None, plot_quantiles_shades: bool = False) -> None:

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
        objective_val, objective_train = retrieve_objective_data(results_dictionary, key)

        percentiles_val = compute_quantiles(objective_val)
        percentiles_train = compute_quantiles(objective_train)

        logger.info(f"The objective for the last generation for key {key} is: {percentiles_val[1][-1]}")

        c = color_list[enumerate_counter]

        plt.plot(np.arange(1, CONFIG["epochs"] + 1, 1), percentiles_val[1], color=c)
        plt.plot(np.arange(1, CONFIG["epochs"] + 1, 1), percentiles_train[1], color=c, linestyle='--')
        legend_entries.append(legend_list[enumerate_counter])
        legend_entries.append(f"_")

        if plot_quantiles_shades:
            plt.fill_between(np.arange(1, CONFIG["epochs"] + 1, 1), percentiles_val[0] + 0.001, percentiles_val[2] - 0.001,
                             color=c, alpha=0.2)
            plt.fill_between(np.arange(1, CONFIG["epochs"] + 1, 1), percentiles_train[0] + 0.001,
                             percentiles_train[2] - 0.001, color=c, alpha=0.2)
            legend_entries.append(f"_")
            legend_entries.append(f"_")

        enumerate_counter += 1

    plt.legend(legend_entries, loc="upper left", fontsize=16)
    plt.yticks(fontsize=16)
    plt.xticks(np.arange(0, CONFIG["epochs"] + 1, 20), fontsize=16)
    plt.xlabel('Generations', size=16)
    plt.ylabel("$\mathcal{F}_{\mathrm{total}}^{1}$", size=17)
    plt.grid(True)

    path_to_save = f"./results/plots/fitness_curves/{dataset_name}/quantiles_plot_bis_{str(plot_quantiles_shades)}.pdf"
    directory = os.path.dirname(path_to_save)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(path_to_save, dpi=300)  # Adjust dpi if needed


def retrieve_transformed_data(results_dictionary: dict, key: str) -> tuple[np.array, np.array, list, list]:

    objective_list = [results_dictionary[key][index][1] for index in range(CONFIG["number_of_runs"])]
    x_transf_train = np.array([objective_list[run][0] for run in range(CONFIG["number_of_runs"])])
    x_transf_val = np.array([objective_list[run][1] for run in range(CONFIG["number_of_runs"])])

    # the size of these can vary by a few instances because of dynamic outlier removal
    x_pca_transf_train = [objective_list[run][2] for run in range(CONFIG["number_of_runs"])]
    x_pca_transf_val = [objective_list[run][3] for run in range(CONFIG["number_of_runs"])]

    return x_transf_train, x_transf_val, x_pca_transf_train, x_pca_transf_val


def plot_2d_scatter(train_data_pca: np.array, val_data_pca: np.array, run_num: int, transformed_or_pca: str, key: str) -> None:

    plt.figure(figsize=(20, 10))
    plt.grid(True)
    plt.scatter(val_data_pca[:, 0], val_data_pca[:, 1], c="red", s=20, edgecolor="k", label="Validation data")
    plt.scatter(train_data_pca[:, 0], train_data_pca[:, 1], c="blue", s=20, edgecolor="k", alpha=0.8, label="Training data")
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.xlabel("$x_1$", size=15)
    plt.ylabel("$x_2$", size=15)
    plt.legend(fontsize=15)

    path_to_save = f"./results/plots/scatter_{transformed_or_pca}/{key}/run_number_{run_num}.pdf"
    directory = os.path.dirname(path_to_save)
    if not os.path.exists(directory):
        os.makedirs(directory)

    plt.savefig(path_to_save, dpi=300)
    plt.close()


if __name__ == "__main__":
    args = parse_arguments()
    dataset = args.dataset

    results_dic = {}
    if dataset in ["spheres", "circles", "alternate_stripes"]:
        base_path = f"./results/synthetic_data/{dataset}/"
        for activation in ["cos", "relu"]:
            for partial_contrib in ["False", "True"]:
                list_runs = []
                file_path = base_path + f"activation={activation}/partial_contrib={partial_contrib}"
                for run_number in range(CONFIG["number_of_runs"]):
                    data = pickle.load(open(file_path+f"/{run_number}.p", "rb"))
                    list_runs.append(data)
                results_dic[file_path] = list_runs

    print(results_dic.keys())
    # save the scatter plot for training and validation
    for k in results_dic.keys():
        x_transformed_train, x_transformed_val, x_pca_transformed_train, x_pca_transformed_val = retrieve_transformed_data(results_dic, k)

        for n_run in range(CONFIG["number_of_runs"]):
            plot_2d_scatter(x_transformed_train[n_run], x_transformed_val[n_run], n_run, "transformed", k)
            plot_2d_scatter(x_pca_transformed_train[n_run], x_pca_transformed_val[n_run], n_run, "pca", k)

    # plot the training and validation loss over the generations
    plot_quantiles(results_dic, dataset, plot_quantiles_shades=True)


