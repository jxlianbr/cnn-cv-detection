"""Sentinel-2 Level-2A surface reflectance composite, autumn 2022, matching
Shimada & Takeuchi (2025)'s acquisition window (reports/paper-draft.md §3.1)."""
import ee
from src.config import Sentinel2Config

S2_COLLECTION = "COPERNICUS/S2_SR_HARMONIZED"
CLOUD_PROB_COLLECTION = "COPERNICUS/S2_CLOUD_PROBABILITY"
CLOUD_PROB_THRESHOLD = 40


def _mask_clouds(image: ee.Image) -> ee.Image:
    cloud_prob = ee.Image(image.get("cloud_probability")).select("probability")
    is_clear = cloud_prob.lt(CLOUD_PROB_THRESHOLD)
    return image.updateMask(is_clear)


def get_autumn_2022_composite(aoi: ee.Geometry, cfg: Sentinel2Config) -> ee.Image:
    s2 = (
        ee.ImageCollection(S2_COLLECTION)
        .filterBounds(aoi)
        .filterDate(cfg.start_date, cfg.end_date)
    )
    cloud_prob = (
        ee.ImageCollection(CLOUD_PROB_COLLECTION)
        .filterBounds(aoi)
        .filterDate(cfg.start_date, cfg.end_date)
    )
    joined = ee.Join.saveFirst("cloud_probability").apply(
        primary=s2, secondary=cloud_prob,
        condition=ee.Filter.equals(leftField="system:index", rightField="system:index"),
    )
    masked = ee.ImageCollection(joined).map(_mask_clouds)
    composite = masked.select(cfg.bands).median().clip(aoi)
    return composite.rename(cfg.bands).multiply(0.0001)  # scale to reflectance [0,1]
