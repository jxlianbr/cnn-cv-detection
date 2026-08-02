import geopandas as gpd
from shapely.geometry import Polygon, box
from src.config import SamplingConfig
from src.data.sampling import generate_sampling_squares, sample_points_in_squares


def _synthetic_pv_gdf():
    polys = [box(139.70 + i * 0.01, 35.60, 139.71 + i * 0.01, 35.61) for i in range(20)]
    return gpd.GeoDataFrame({"geometry": polys}, crs="EPSG:4326")


def _synthetic_buffer_gdf():
    return gpd.GeoDataFrame({"geometry": [box(139.5, 35.4, 140.0, 35.9)]}, crs="EPSG:4326")


def _cfg():
    return SamplingConfig(
        n_sampling_squares=100, pv_squares=60, buffer_squares=30,
        urban_squares=10, buffer_km=7,
    )


def test_generate_sampling_squares_respects_stratum_counts():
    squares = generate_sampling_squares(
        _synthetic_pv_gdf(), _synthetic_buffer_gdf(), _cfg(), seed=42
    )
    assert len(squares) == 100
    assert (squares["stratum"] == "pv").sum() == 60
    assert (squares["stratum"] == "buffer").sum() == 30
    assert (squares["stratum"] == "urban").sum() == 10


def test_generate_sampling_squares_pv_squares_intersect_pv_polygons():
    pv_gdf = _synthetic_pv_gdf()
    squares = generate_sampling_squares(pv_gdf, _synthetic_buffer_gdf(), _cfg(), seed=42)
    pv_squares = squares[squares["stratum"] == "pv"]
    union = pv_gdf.geometry.union_all()
    assert all(sq.intersects(union) for sq in pv_squares.geometry)


def test_generate_sampling_squares_is_deterministic_given_seed():
    a = generate_sampling_squares(_synthetic_pv_gdf(), _synthetic_buffer_gdf(), _cfg(), seed=1)
    b = generate_sampling_squares(_synthetic_pv_gdf(), _synthetic_buffer_gdf(), _cfg(), seed=1)
    # GeoSeries.equals is pandas' scalar comparison; geom_equals is the
    # element-wise geometry comparison this test actually means.
    assert a.geometry.geom_equals(b.geometry).all()


def test_sample_points_in_squares_produces_requested_counts_and_labels():
    squares = generate_sampling_squares(
        _synthetic_pv_gdf(), _synthetic_buffer_gdf(), _cfg(), seed=42
    )
    points = sample_points_in_squares(
        squares, _synthetic_pv_gdf(), n_pv=50, n_nonpv=100, seed=42
    )
    assert (points["label"] == "pv").sum() == 50
    assert (points["label"] == "non_pv").sum() == 100
    assert points.geometry.geom_type.eq("Point").all()


def test_sample_points_pv_points_fall_inside_pv_polygons():
    pv_gdf = _synthetic_pv_gdf()
    squares = generate_sampling_squares(pv_gdf, _synthetic_buffer_gdf(), _cfg(), seed=42)
    points = sample_points_in_squares(squares, pv_gdf, n_pv=20, n_nonpv=20, seed=42)
    pv_points = points[points["label"] == "pv"]
    union = pv_gdf.geometry.union_all()
    assert all(p.within(union) for p in pv_points.geometry)
