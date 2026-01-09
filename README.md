Non-linear PCA by training neural networks using Evolution Strategies (ES).

We optimize a set of feature transformations so that, after transforming the data, PCA maximizes explained variance along the top components. The method natively supports numerical, categorical, and ordinal variables, as well as mixed-type datasets (categoricals are automatically one-hot encoded where appropriate).


## Requirements
- Python 3.10

Main runtime dependencies (managed via Poetry): numpy, pandas, scipy, scikit-learn, pydantic, pyyaml, matplotlib, loguru, jupyter.


## Installation

Recommended (Poetry):
- Install Poetry if you don’t have it: https://python-poetry.org/docs/#installation
- From the repository root:
  - poetry install
  - poetry run python --version (should be 3.10.x)

Alternative (pip, without Poetry):
- Create and activate a virtual environment using your preferred tool
- Install the package in editable mode and dependencies:
  - pip install -e .
  - pip install numpy pandas scipy scikit-learn pydantic pyyaml matplotlib loguru jupyter

Note: All commands below assume you run them from the repository root.


## Quick start: Run PCA-ES

Run directly via Python:

```
python es_pca/main.py --dataset <dataset> --partial_contrib <true|false> --activation <relu|tanh|...>
```

Or run the provided convenience script (edits inside the file control the grid of runs):

```
./run_main.sh
```

The script and CLI will read default values from config_es.yaml and datasets_config.yaml (see below). Command-line flags override the corresponding defaults.


## Configuration

1) Evolution Strategies and model settings (config_es.yaml)
- Controls ES parameters and the neural network architecture/training loop.
- Key fields (non-exhaustive):
  - hidden_layer_size, n_hidden_layers, activation, batch_norm
  - pop_size, sigma, learning_rate, epochs, batch_size, early_stopping_epochs
  - pca_type, partial_contribution_objective, num_components
  - dataset, number_of_runs, val_prop

2) Dataset schema and preprocessing (datasets_config.yaml)
- For each dataset name, define the column types (e.g., which columns are categorical).
- Categorical features are automatically one-hot encoded before training.


## Datasets and formatting

- Real-world datasets are expected under datasets/<name>.arff
- Synthetic datasets are generated on the fly (supported: circles, spheres, alternate_stripes) — no files required.

The datasets_config.yaml file must include an entry for the dataset you want to run. For example (illustrative snippet):

```
my_dataset:
  categorical_features: [0, 3]  # indices of categorical columns in the input table
  # other dataset-specific options can be defined here
```

When categorical_features is provided, those columns will be encoded to one-hot internally, and the model will handle mixed data types accordingly.


## Results and outputs

During training, results are saved under:

```
results/datasets/{synthetic_data|real_world_data}/<dataset>/
  k=<num_components>/
  activation=<activation>/
  partial_contrib=<true|false>/<run_index>/
```

Artifacts can include:
- best_individual_epoch_*.p: serialized best solutions at checkpoints
- results_list.p: list of results accumulated during training
- Plots (if enabled via config_es.yaml: plot: true)

To analyze or summarize results, you can use:

```
./run_read_results.sh -d <dataset> -p <true|false> -k <num_components>
```

Examples:

```
# Read results for ionosphere with partial contribution and k=2
./run_read_results.sh -d ionosphere -p true -k 2

# Read results for circles without partial contribution and k=3
./run_read_results.sh --dataset circles --partial_contrib false --num_components 3
```

This script forwards the selected dataset, partial_contrib, and PCA dimensionality k to es_pca/read_results.py, which filters runs in the corresponding results/datasets/.../k=<num_components>/ folder.

You can also invoke the Python reader directly:

```
python es_pca/read_results.py --dataset ionosphere --partial_contrib true --num_components 2
```


## Reproducibility and multiple runs

The number_of_runs field in config_es.yaml controls how many independent runs are launched. By default, main.py will distribute these over available CPU cores using Python’s multiprocessing.


## Tips and troubleshooting
- Always run commands from the repository root so relative paths (e.g., datasets_config.yaml, results/) resolve correctly.
- Ensure your dataset name exists in datasets_config.yaml and, for real datasets, that datasets/<name>.arff is present.
- partial_contrib is a string flag on the CLI: use "true" or "false" (lowercase). It is converted to boolean internally.
- For standardized preprocessing of most real datasets, the code applies a StandardScaler on the training split and uses it on the validation split.


## Citation
If you use this project in academic work, please cite it appropriately. A formal citation will be added here when available.