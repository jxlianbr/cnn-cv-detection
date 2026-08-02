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


def test_sample_points_fills_pv_quota_when_pv_polygons_are_far_apart():
    # Real PV is scattered across the AOI, so the union's bounding box is
    # enormous relative to the polygons themselves. Rejection sampling over
    # that bbox almost never lands inside a polygon and the pv class starves.
    squares = gpd.GeoDataFrame(
        {"stratum": ["pv", "pv"],
         "geometry": [box(139.000, 35.000, 139.009, 35.009),
                      box(140.000, 36.000, 140.009, 36.009)]},
        crs="EPSG:4326",
    )
    pv_gdf = gpd.GeoDataFrame(
        {"geometry": [box(139.001, 35.001, 139.002, 35.002),
                      box(140.001, 36.001, 140.002, 36.002)]},
        crs="EPSG:4326",
    )

    points = sample_points_in_squares(squares, pv_gdf, n_pv=50, n_nonpv=20, seed=0)

    assert (points["label"] == "pv").sum() == 50
    assert (points["label"] == "non_pv").sum() == 20


def test_sample_points_covers_every_disjoint_pv_polygon():
    # a sampler that only ever finds the first polygon would still fill the
    # quota; the sample has to reach both installations
    squares = gpd.GeoDataFrame(
        {"stratum": ["pv", "pv"],
         "geometry": [box(139.000, 35.000, 139.009, 35.009),
                      box(140.000, 36.000, 140.009, 36.009)]},
        crs="EPSG:4326",
    )
    far_a = box(139.001, 35.001, 139.002, 35.002)
    far_b = box(140.001, 36.001, 140.002, 36.002)
    pv_gdf = gpd.GeoDataFrame({"geometry": [far_a, far_b]}, crs="EPSG:4326")

    points = sample_points_in_squares(squares, pv_gdf, n_pv=60, n_nonpv=5, seed=0)
    pv_points = points[points["label"] == "pv"]

    assert any(p.within(far_a) for p in pv_points.geometry)
    assert any(p.within(far_b) for p in pv_points.geometry)


def test_sample_points_pv_points_fall_inside_pv_polygons():
    pv_gdf = _synthetic_pv_gdf()
    squares = generate_sampling_squares(pv_gdf, _synthetic_buffer_gdf(), _cfg(), seed=42)
    points = sample_points_in_squares(squares, pv_gdf, n_pv=20, n_nonpv=20, seed=42)
    pv_points = points[points["label"] == "pv"]
    union = pv_gdf.geometry.union_all()
    assert all(p.within(union) for p in pv_points.geometry)
