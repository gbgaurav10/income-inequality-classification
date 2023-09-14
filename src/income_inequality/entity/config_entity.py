
import os, sys
from dataclasses import dataclass
from pathlib import Path 

@dataclass(frozen=True)
class DataIngestionConfig:
    root_dir: Path 
    source_url: str 
    local_data_file: Path 
    unzip_dir: Path


@dataclass(frozen=True)
class DataTransformationConfig:
    root_dir: Path
    data_path: Path
    preprocessor_path: Path


@dataclass(frozen=True)
class ModelTrainingConfig:
    root_dir: Path 
    train_data_path: Path 
    test_data_path: Path 
    model_name: str 
    n_estimators: int 
    max_depth: int 
    criterion: str
    target_column: str 