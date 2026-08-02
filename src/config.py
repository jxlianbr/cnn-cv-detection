from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import yaml


@dataclass
class Sentinel2Config:
    start_date: str
    end_date: str
    bands: list[str]


@dataclass
class XGBoostConfig:
    max_depth: int
    eta: float
    n_estimators: int


@dataclass
class SamplingConfig:
    n_sampling_squares: int
    pv_squares: int
    buffer_squares: int
    urban_squares: int
    buffer_km: float


@dataclass
class PatchConfig:
    size_px: int


@dataclass
class PathsConfig:
    pv_polygons: str


@dataclass
class Config:
    gee_project_id: str
    sentinel2: Sentinel2Config
    xgboost: XGBoostConfig
    sampling: SamplingConfig
    patch: PatchConfig
    paths: PathsConfig


def load_config(path: str | Path = "configs/config.yaml") -> Config:
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return Config(
        gee_project_id=raw["gee_project_id"],
        sentinel2=Sentinel2Config(**raw["sentinel2"]),
        xgboost=XGBoostConfig(**raw["xgboost"]),
        sampling=SamplingConfig(**raw["sampling"]),
        patch=PatchConfig(**raw["patch"]),
        paths=PathsConfig(**raw["paths"]),
    )
