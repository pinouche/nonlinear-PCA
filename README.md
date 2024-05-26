Non-linear PCA by training neural networks using Evolution Strategies (ES). The ES process is optimized by finding transformations for which the explained variance is maximized when applying PCA on the transformations. In addition to numerical variables, our method also natively supports categorical variables,  ordinal variables, and a mix of the three.


# Run PCA-ES

This script can be run directly with:

```
python3 src/main.py 
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

