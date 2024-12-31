Non-linear PCA by training neural networks using Evolution Strategies (ES). The ES process is optimized by finding transformations for which the explained variance is maximized when applying PCA on the transformations. In addition to numerical variables, our method also natively supports categorical variables,  ordinal variables, and a mix of the three.


# Run PCA-ES

This script can be run directly with:

```
python es_pca/main.py --dataset <dataset> --partial_contrib <partial_contrib> --activation <activation>
```

or, alternatively, one can run the provided bash file:

```
./run_main.sh
```

The ```config_es.yaml``` contains all the parameters to fully specify a ES run (ES parameters, neural networks parameters).

# Run the collating of the results (plotting)


```
python3 es_pca/read_results.py --dataset <dataset>
```

# Formatting of the datasets

The column types of the dataset must be defined in the datasets_config.yaml file. Each categorical variable will then be
automatically converted to one-hot and use the appropriate transformations.

# Datasets to perform experiments on:

- circles: done
- spheres: (running)
- wine: (to re-run on 500 epochs)
- heart-statlog: done