from src.config import load_config


def test_load_config_returns_expected_fields(tmp_path):
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text(
        """
gee_project_id: "my-test-project"
sentinel2:
  start_date: "2022-09-01"
  end_date: "2022-11-30"
  bands: [B2, B3, B4, B8, B11, B12]
xgboost:
  max_depth: 8
  eta: 0.01717
  n_estimators: 1000
sampling:
  n_sampling_squares: 1000
  pv_squares: 600
  buffer_squares: 300
  urban_squares: 100
  buffer_km: 7
patch:
  size_px: 21
aoi:
  name: "kanto_pilot"
  bbox: [139.0, 35.3, 140.3, 36.3]
split:
  test_fraction: 0.2
  val_fraction: 0.2
  block_size_km: 10
  seed: 42
paths:
  pv_polygons: "data/external/pv_polygons.gpkg"
  processed_dir: "data/processed"
"""
    )
    cfg = load_config(cfg_yaml)
    assert cfg.gee_project_id == "my-test-project"
    assert cfg.sentinel2.bands == ["B2", "B3", "B4", "B8", "B11", "B12"]
    assert cfg.xgboost.max_depth == 8
    assert cfg.xgboost.eta == 0.01717
    assert cfg.sampling.pv_squares == 600
    assert cfg.patch.size_px == 21
    assert cfg.aoi.name == "kanto_pilot"
    assert cfg.aoi.bbox == [139.0, 35.3, 140.3, 36.3]
    assert cfg.split.test_fraction == 0.2
    assert cfg.split.block_size_km == 10
    assert cfg.split.seed == 42
    assert cfg.paths.processed_dir == "data/processed"
