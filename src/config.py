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
class AOIConfig:
    name: str
    bbox: list[float]  # [west, south, east, north] in EPSG:4326


@dataclass
class SplitConfig:
    test_fraction: float
    val_fraction: float
    block_size_km: float
    seed: int


@dataclass
class PathsConfig:
    pv_polygons: str
    processed_dir: str


@dataclass
class Config:
    gee_project_id: str
    sentinel2: Sentinel2Config
    xgboost: XGBoostConfig
    sampling: SamplingConfig
    patch: PatchConfig
    aoi: AOIConfig
    split: SplitConfig
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
        aoi=AOIConfig(**raw["aoi"]),
        split=SplitConfig(**raw["split"]),
        paths=PathsConfig(**raw["paths"]),
    )
