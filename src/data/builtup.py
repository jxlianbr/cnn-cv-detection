"""Built-up-area proxy for the original study's manually delineated
"populated areas" / "urban" sampling strata, using the JRC Global Human
Settlement Layer (GHSL) built-up surface product in Earth Engine."""
import ee

GHSL_BUILT_UP_IMAGE = "JRC/GHSL/P2023A/GHS_BUILT_S/2020"


def _built_up_mask(percentile_image: ee.Image, threshold: float) -> ee.Image:
    return percentile_image.gt(threshold)


def get_populated_buffer(aoi: ee.Geometry, buffer_km: float,
                         built_up_threshold_m2: float = 1000) -> ee.FeatureCollection:
    """Vectorized buffer around GHSL-flagged built-up pixels within aoi."""
    built_up = ee.Image(GHSL_BUILT_UP_IMAGE).select("built_surface").clip(aoi)
    mask = _built_up_mask(built_up, built_up_threshold_m2)
    vectors = mask.selfMask().reduceToVectors(
        geometry=aoi, scale=100, geometryType="polygon", maxPixels=1e10
    )
    buffered = vectors.map(lambda f: f.buffer(buffer_km * 1000))
    return buffered


def sample_urban_core_points(aoi: ee.Geometry, n_points: int,
                             built_up_threshold_m2: float = 5000) -> ee.FeatureCollection:
    """Random points within the densest built-up pixels in aoi, as a proxy
    for the original study's manually selected urban non-PV samples."""
    built_up = ee.Image(GHSL_BUILT_UP_IMAGE).select("built_surface").clip(aoi)
    dense_core = _built_up_mask(built_up, built_up_threshold_m2).selfMask()
    return dense_core.stratifiedSample(
        numPoints=n_points, classBand="built_surface", region=aoi,
        scale=100, geometries=True,
    )
