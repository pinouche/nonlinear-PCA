from pydantic import BaseModel


class DatasetModel(BaseModel):

    dataset: str
    num_features: int
    synthetic_bool: bool
    contains_categorical: bool