Non-linear PCA by training neural networks using Evolution Strategies (ES). The ES process is optimized by finding transformations for which the explained variance is maximized when applying PCA on the transformations. In addition to numerical variables, our method also natively supports categorical variables,  ordinal variables, and a mix of the three.

This script is run using:

```
python3 src/main.py 
```

where the ```config_es.yaml``` contains all the parameters to fully specify a ES run (ES parameters, neural networks parameters).

