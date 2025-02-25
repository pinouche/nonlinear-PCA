# Define the arrays for datasets, partial_contribution_objective values, activation values, and batch normalization values
datasets=("german_credit")
partial_contrib_values=("false")
activation_values=("relu")
batch_norm_values=("true")

# Loop over each combination of dataset, partial_contrib, activation, and batch_norm
for dataset in "${datasets[@]}"; do
  for partial_contrib in "${partial_contrib_values[@]}"; do
    for activation in "${activation_values[@]}"; do
      for batch_norm in "${batch_norm_values[@]}"; do
        # Run the Python script with the current combination of arguments
        python es_pca/main.py --dataset "$dataset" --partial_contrib "$partial_contrib" --activation "$activation" --batch_norm "$batch_norm"
      done
    done
  done
done