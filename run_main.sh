# Define the arrays for datasets, partial_contribution_objective values, and activation values
datasets=("wine")
partial_contrib_values=("true" "false")
activation_values=("relu")

# Loop over each combination of dataset, partial_contrib, and activation
for dataset in "${datasets[@]}"; do
  for partial_contrib in "${partial_contrib_values[@]}"; do
    for activation in "${activation_values[@]}"; do
      # Run the Python script with the current combination of arguments
      python es_pca/main.py --dataset "$dataset" --partial_contrib "$partial_contrib" --activation "$activation"
    done
  done
done