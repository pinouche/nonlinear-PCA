#!/usr/bin/env bash

# Define the arrays for datasets, partial_contribution_objective values, activation values, and number of principal components
datasets=("circles")
partial_contrib_values=("true" "false")
activation_values=("relu")
num_components_values=(1)

# Loop over each combination of dataset, partial_contrib, activation, and batch_norm
for dataset in "${datasets[@]}"; do
  for partial_contrib in "${partial_contrib_values[@]}"; do
    for activation in "${activation_values[@]}"; do
      for k in "${num_components_values[@]}"; do
        # Run the Python script with the current combination of arguments
        python es_pca/main.py \
          --dataset "$dataset" \
          --partial_contrib "$partial_contrib" \
          --activation "$activation" \
          --num_components "$k"
      done
    done
  done
done