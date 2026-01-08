from pydantic import BaseModel, Field
from typing import List, Optional


class ConfigDataset(BaseModel):
    num_features: int
    synthetic_bool: bool
    categorical_features: Optional[List[int]] = Field(default_factory=list)
    numerical_features: Optional[List[int]] = Field(default_factory=list)


class ConfigES(BaseModel):
    # Neural network + ES hyperparameters
    hidden_layer_size: int
    activation: str
    batch_norm: bool
    init_mode: str
    n_hidden_layers: int
    pop_size: int
    sigma: float
    learning_rate: float
    epochs: int
    batch_size: int
    early_stopping_epochs: int

    # PCA and objective settings
    alpha_reg_pca: float
    pca_type: str
    partial_contribution_objective: bool
    num_components: int

    # Run setup
    dataset: str
    number_of_runs: int
    remove_outliers: bool
    n_outliers: int
    plot: bool
    val_prop: float

