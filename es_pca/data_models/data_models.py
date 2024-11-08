from pydantic import BaseModel, Field
from typing import List, Optional


class ConfigDataset(BaseModel):
    num_features: int
    synthetic_bool: bool
    categorical_features: Optional[List[int]] = Field(default_factory=list)
    numerical_features: Optional[List[int]] = Field(default_factory=list)

