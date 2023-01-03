Non-linear PCA by training neural networks using Evolution Strategies (ES). The ES process is optimized by finding transformations for which the explained variance is maxized when applying PCA on the transformations. In addition to numerical variables, our method also natively supports categorical variables,  ordinal variables, and a mix of the three.

This script is run using:

```
python3 main.py --filename config.json5
```

where the ```config.json5``` contains all the parameters to fully specify a ES run (ES parameters, neural networks parameters).

