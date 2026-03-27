"""
Debug tests for SkeletonExtended, GriddedSkeletonExtended, PointSkeletonExtended.
Uses print statements to trace what is happening at each step.
"""

import numpy as np
import pytest
import warnings
from pyproj import CRS

# --- adjust these imports to match your actual module paths ---
from geo_skeletons.skeleton_extended import SkeletonExtended
from geo_skeletons.gridded_skeleton_extended import GriddedSkeletonExtended
from geo_skeletons.point_skeleton_extended import (
    PointSkeletonExtended,
)  # adjust if named differently

UTM33N = CRS.from_epsg(32633)
ROTATED_POLE = CRS.from_cf(
    {
        "grid_mapping_name": "rotated_latitude_longitude",
        "grid_north_pole_latitude": 39.25,
        "grid_north_pole_longitude": -162.0,
    }
)

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def make_spherical(nx=5, ny=4):
    g = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    g.set_spacing(nx=nx, ny=ny)
    return g


def make_cartesian(nx=5, ny=4):
    g = GriddedSkeletonExtended(x=(300_000, 700_000), y=(6_500_000, 7_000_000))
    g.set_spacing(nx=nx, ny=ny)
    g.proj.set(UTM33N, silent=True)
    return g


def make_rotated(nx=5, ny=4):
    g = GriddedSkeletonExtended(rlon=(-5.0, 5.0), rlat=(-3.0, 3.0))
    g.set_spacing(nx=nx, ny=ny)
    g.proj.set(ROTATED_POLE, silent=True)
    return g


# ------------------------------------------------------------------
# 1. Basic construction — does __init__ complete without error?
# ------------------------------------------------------------------


def test_gridded_spherical_init():
    print("\n--- test_gridded_spherical_init ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    print(f"  ds():        {grid.ds()}")
    print(f"  nx():        {grid.nx()}")
    print(f"  ny():        {grid.ny()}")
    print(f"  core.x_str:  {grid.core.x_str}")
    print(f"  has proj:    {hasattr(grid, 'proj')}")
    print(f"  proj:        {grid.proj.projection()}")
    assert grid.nx() == 2
    assert grid.ny() == 2


def test_gridded_cartesian_init():
    print("\n--- test_gridded_cartesian_init ---")
    grid = GriddedSkeletonExtended(x=(300_000, 700_000), y=(6_500_000, 7_000_000))
    print(f"  ds():        {grid.ds()}")
    print(f"  nx():        {grid.nx()}")
    print(f"  ny():        {grid.ny()}")
    print(f"  core.x_str:  {grid.core.x_str}")
    print(f"  has proj:    {hasattr(grid, 'proj')}")
    assert grid.nx() == 2
    assert grid.ny() == 2


def test_point_spherical_init():
    print("\n--- test_point_spherical_init ---")
    pts = PointSkeletonExtended(lon=[5.0, 6.0, 7.0], lat=[58.0, 59.0, 60.0])
    print(f"  ds():        {pts.ds()}")
    print(f"  nx():        {pts.nx()}")
    print(f"  has proj:    {hasattr(pts, 'proj')}")
    assert pts.nx() == 3


def test_point_cartesian_init():
    print("\n--- test_point_cartesian_init ---")
    pts = PointSkeletonExtended(x=[0.0, 1.0, 2.0], y=[0.0, 1.0, 2.0])
    print(f"  ds():        {pts.ds()}")
    print(f"  nx():        {pts.nx()}")
    print(f"  has proj:    {hasattr(pts, 'proj')}")
    assert pts.nx() == 3


def test_rotated_init():
    print("\n--- test_rotated_init ---")
    g = GriddedSkeletonExtended(rlon=(-5.0, 5.0), rlat=(-3.0, 3.0))
    print(f"  core.x_str: {g.core.x_str}")
    print(f"  core.y_str: {g.core.y_str}")
    print(f"  is_rotated: {g.core.is_rotated()}")
    print(f"  is_cartesian: {g.core.is_cartesian()}")
    print(f"  is_spherical: {g.core.is_spherical()}")
    assert g.core.is_rotated()
    assert not g.core.is_cartesian()
    assert not g.core.is_spherical()
    assert g.nx() == 2
    assert g.ny() == 2


def test_rotated_rlon_rlat_stored():
    print("\n--- test_rotated_rlon_rlat_stored ---")
    g = make_rotated()
    rlon = g.rlon()
    rlat = g.rlat()
    print(f"  rlon: {rlon}")
    print(f"  rlat: {rlat}")
    assert rlon is not None
    assert rlat is not None
    np.testing.assert_array_almost_equal(rlon, np.linspace(-5.0, 5.0, 5))
    np.testing.assert_array_almost_equal(rlat, np.linspace(-3.0, 3.0, 4))


# ------------------------------------------------------------------
# 2. proj manager — is it set up correctly?
# ------------------------------------------------------------------


def test_proj_manager_exists_after_init():
    print("\n--- test_proj_manager_exists_after_init ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    print(f"  type(grid.proj): {type(grid.proj)}")
    print(f"  grid.proj.projection(): {grid.proj.projection()}")
    assert hasattr(grid, "proj")


def test_proj_set_utm33n():
    print("\n--- test_proj_set_utm33n ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    grid.proj.set(UTM33N, silent=True)
    print(
        f"  projection after set: {grid.proj.projection()}, {grid.proj.projection().name}"
    )
    assert grid.proj.projection() is not None


def test_no_utm_attribute():
    print("\n--- test_no_utm_attribute ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    has_utm = hasattr(grid, "utm")
    print(f"  has 'utm' attribute: {has_utm}  (should be False)")
    assert not has_utm


# ------------------------------------------------------------------
# 3. lon/lat/x/y on spherical GriddedSkeleton
# ------------------------------------------------------------------


def test_gridded_spherical_lon_lat():
    print("\n--- test_gridded_spherical_lon_lat ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    grid.set_spacing(nx=5, ny=4)
    lon = grid.lon()
    lat = grid.lat()
    print(f"  lon: {lon}")
    print(f"  lat: {lat}")
    print(f"  lon shape: {lon.shape}")
    print(f"  lat shape: {lat.shape}")
    assert lon is not None
    assert lat is not None
    np.testing.assert_array_almost_equal(lon, np.linspace(5.0, 15.0, 5))
    np.testing.assert_array_almost_equal(lat, np.linspace(58.0, 65.0, 4))


def test_gridded_spherical_x_strict_returns_none():
    print("\n--- test_gridded_spherical_x_strict_returns_none ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    x = grid.x(strict=True)
    print(f"  x(strict=True): {x}  (should be None)")
    assert x is None


def test_gridded_spherical_x_with_proj():
    print("\n--- test_gridded_spherical_x_with_proj ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    grid.set_spacing(nx=5, ny=4)
    grid.proj.set(UTM33N, silent=True)
    x = grid.x()
    y = grid.y()
    print(f"  x range: [{x.min():.0f}, {x.max():.0f}]")
    print(f"  y range: [{y.min():.0f}, {y.max():.0f}]")
    print(f"  x(): {x}")
    print(f"  y(): {y}")
    assert x is not None
    assert y is not None


# ------------------------------------------------------------------
# 4. lon/lat/x/y on cartesian GriddedSkeleton
# ------------------------------------------------------------------


def test_gridded_cartesian_x_y():
    print("\n--- test_gridded_cartesian_x_y ---")
    grid = GriddedSkeletonExtended(x=(300_000, 700_000), y=(6_500_000, 7_000_000))
    grid.set_spacing(nx=5, ny=4)
    x = grid.x()
    y = grid.y()
    print(f"  x: {x}")
    print(f"  y: {y}")
    np.testing.assert_array_almost_equal(x, np.linspace(300_000, 700_000, 5))
    np.testing.assert_array_almost_equal(y, np.linspace(6_500_000, 7_000_000, 4))


def test_gridded_cartesian_lon_strict_returns_none():
    print("\n--- test_gridded_cartesian_lon_strict_returns_none ---")
    grid = GriddedSkeletonExtended(x=(300_000, 700_000), y=(6_500_000, 7_000_000))
    lon = grid.lon(strict=True)
    print(f"  lon(strict=True): {lon}  (should be None)")
    assert lon is None


def test_gridded_cartesian_lon_with_proj():
    print("\n--- test_gridded_cartesian_lon_with_proj ---")
    grid = GriddedSkeletonExtended(x=(300_000, 700_000), y=(6_500_000, 7_000_000))
    grid.set_spacing(nx=5, ny=4)
    grid.proj.set(UTM33N, silent=True)
    lon = grid.lon()
    lat = grid.lat()
    print(f"  lon: {lon}")
    print(f"  lat: {lat}")
    assert lon is not None
    assert lat is not None
    assert np.all(lon >= -180) and np.all(lon <= 180)
    assert np.all(lat >= -90) and np.all(lat <= 90)


# ------------------------------------------------------------------
# 5. lonlat() and xy() — all points
# ------------------------------------------------------------------


def test_gridded_lonlat_returns_all_points():
    print("\n--- test_gridded_lonlat_returns_all_points ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    grid.set_spacing(nx=5, ny=4)
    lon, lat = grid.lonlat()
    print(f"  lon shape: {lon.shape}  (should be 20,)")
    print(f"  lat shape: {lat.shape}  (should be 20,)")
    assert lon.shape == (20,)
    assert lat.shape == (20,)


def test_gridded_cartesian_xy_returns_all_points():
    print("\n--- test_gridded_cartesian_xy_returns_all_points ---")
    grid = GriddedSkeletonExtended(x=(0, 4), y=(0, 3))
    grid.set_spacing(nx=5, ny=4)
    x, y = grid.xy()
    print(f"  x shape: {x.shape}  (should be 20,)")
    print(f"  y shape: {y.shape}  (should be 20,)")
    assert x.shape == (20,)
    assert y.shape == (20,)


# ------------------------------------------------------------------
# 6. set_spacing
# ------------------------------------------------------------------


def test_set_spacing_nx_ny():
    print("\n--- test_set_spacing_nx_ny ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    grid.set_spacing(nx=10, ny=8)
    print(f"  nx: {grid.nx()}  (should be 10)")
    print(f"  ny: {grid.ny()}  (should be 8)")
    assert grid.nx() == 10
    assert grid.ny() == 8


def test_set_spacing_dlon_dlat():
    print("\n--- test_set_spacing_dlon_dlat ---")
    grid = GriddedSkeletonExtended(lon=(0.0, 1.0), lat=(0.0, 1.0))
    grid.set_spacing(dlon=0.5, dlat=0.5)
    print(f"  nx: {grid.nx()}  (should be 3)")
    print(f"  ny: {grid.ny()}  (should be 3)")
    assert grid.nx() == 3
    assert grid.ny() == 3


# ------------------------------------------------------------------
# 7. proj conversion round-trip
# ------------------------------------------------------------------


def test_lonlat_to_xy_roundtrip():
    print("\n--- test_lonlat_to_xy_roundtrip ---")
    grid = GriddedSkeletonExtended(lon=(5.0, 15.0), lat=(58.0, 65.0))
    grid.set_spacing(nx=5, ny=4)
    grid.proj.set(UTM33N, silent=True)

    lon_orig, lat_orig = grid.lonlat()
    x, y = grid.xy()
    print(f"  x range: [{x.min():.0f}, {x.max():.0f}]")
    print(f"  y range: [{y.min():.0f}, {y.max():.0f}]")

    # Convert back via proj
    lon_back = grid.proj._lon(x=x, y=y)
    lat_back = grid.proj._lat(x=x, y=y)
    print(f"  max lon diff: {np.max(np.abs(lon_orig - lon_back)):.6f} deg")
    print(f"  max lat diff: {np.max(np.abs(lat_orig - lat_back)):.6f} deg")
    np.testing.assert_array_almost_equal(lon_orig, lon_back, decimal=4)
    np.testing.assert_array_almost_equal(lat_orig, lat_back, decimal=4)


# ------------------------------------------------------------------
# rlon/rlat return None for non-rotated grids
# ------------------------------------------------------------------


def test_rlon_returns_none_for_spherical():
    print("\n--- test_rlon_returns_none_for_spherical ---")
    g = make_spherical()
    print(f"  rlon: {g.rlon()}")
    assert g.rlon() is None


def test_rlon_returns_none_for_cartesian():
    print("\n--- test_rlon_returns_none_for_cartesian ---")
    g = make_cartesian()
    print(f"  rlon: {g.rlon()}")
    assert g.rlon() is None


# ------------------------------------------------------------------
# lon/lat on rotated grid
# ------------------------------------------------------------------


def test_rotated_lon_returns_true_geographic():
    print("\n--- test_rotated_lon_returns_true_geographic ---")
    g = make_rotated()
    lon = g.lon()
    print(f"  lon shape: {lon.shape}")
    print(f"  lon range: [{lon.min():.3f}, {lon.max():.3f}]")
    assert lon is not None
    assert np.all(lon >= -180) and np.all(lon <= 180)


def test_rotated_lat_returns_true_geographic():
    print("\n--- test_rotated_lat_returns_true_geographic ---")
    g = make_rotated()
    lat = g.lat()
    print(f"  lat shape: {lat.shape}")
    print(f"  lat range: [{lat.min():.3f}, {lat.max():.3f}]")
    assert lat is not None
    assert np.all(lat >= -90) and np.all(lat <= 90)


def test_rotated_lon_strict_returns_none():
    print("\n--- test_rotated_lon_strict_returns_none ---")
    g = make_rotated()
    lon = g.lon(strict=True)
    print(f"  lon(strict=True): {lon}")
    assert lon is None


def test_rotated_lat_strict_returns_none():
    print("\n--- test_rotated_lat_strict_returns_none ---")
    g = make_rotated()
    lat = g.lat(strict=True)
    print(f"  lat(strict=True): {lat}")
    assert lat is None


def test_rotated_lon_native_warns_and_returns_rlon():
    print("\n--- test_rotated_lon_native_warns_and_returns_rlon ---")
    g = make_rotated()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lon = g.lon(native=True)
        print(f"  warning: {w[0].message if w else 'none'}")
        print(f"  lon: {lon}")
        assert len(w) == 1
        assert issubclass(w[0].category, UserWarning)
    np.testing.assert_array_almost_equal(lon, g.rlon())


def test_rotated_lat_native_warns_and_returns_rlat():
    print("\n--- test_rotated_lat_native_warns_and_returns_rlat ---")
    g = make_rotated()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lat = g.lat(native=True)
        print(f"  warning: {w[0].message if w else 'none'}")
        assert len(w) == 1
    np.testing.assert_array_almost_equal(lat, g.rlat())


# ------------------------------------------------------------------
# x/y on rotated grid raises
# ------------------------------------------------------------------


def test_rotated_x_raises():
    print("\n--- test_rotated_x_raises ---")
    g = make_rotated()
    with pytest.raises(ValueError, match="rlon"):
        g.x()


def test_rotated_y_raises():
    print("\n--- test_rotated_y_raises ---")
    g = make_rotated()
    with pytest.raises(ValueError, match="rlat"):
        g.y()


def test_rotated_x_strict_returns_none():
    print("\n--- test_rotated_x_strict_returns_none ---")
    g = make_rotated()
    assert g.x(strict=True) is None


def test_rotated_y_strict_returns_none():
    print("\n--- test_rotated_y_strict_returns_none ---")
    g = make_rotated()
    assert g.y(strict=True) is None


def test_rotated_x_native_returns_rlon():
    print("\n--- test_rotated_x_native_returns_rlon ---")
    g = make_rotated()
    x = g.x(native=True)
    print(f"  x(native=True): {x}")
    np.testing.assert_array_almost_equal(x, g.rlon())


def test_rotated_y_native_returns_rlat():
    print("\n--- test_rotated_y_native_returns_rlat ---")
    g = make_rotated()
    y = g.y(native=True)
    print(f"  y(native=True): {y}")
    np.testing.assert_array_almost_equal(y, g.rlat())


# ------------------------------------------------------------------
# rlonlat / rlongrid / rlatgrid
# ------------------------------------------------------------------


def test_rotated_rlonlat_shape():
    print("\n--- test_rotated_rlonlat_shape ---")
    g = make_rotated(nx=5, ny=4)
    rlon, rlat = g.rlonlat()
    print(f"  rlon shape: {rlon.shape}  (should be 20,)")
    print(f"  rlat shape: {rlat.shape}  (should be 20,)")
    assert rlon.shape == (20,)
    assert rlat.shape == (20,)


def test_rlonlat_returns_none_for_spherical():
    print("\n--- test_rlonlat_returns_none_for_spherical ---")
    g = make_spherical()
    rlon, rlat = g.rlonlat()
    print(f"  rlonlat: {rlon}, {rlat}")
    assert rlon is None
    assert rlat is None


def test_rlonlat_returns_none_for_cartesian():
    print("\n--- test_rlonlat_returns_none_for_cartesian ---")
    g = make_cartesian()
    rlon, rlat = g.rlonlat()
    assert rlon is None
    assert rlat is None


def test_rotated_rlonlat_values_match_rlon_rlat():
    print("\n--- test_rotated_rlonlat_values_match_rlon_rlat ---")
    g = make_rotated(nx=5, ny=4)
    rlon_flat, rlat_flat = g.rlonlat()
    expected_rlon, expected_rlat = np.meshgrid(g.rlon(), g.rlat())
    print(f"  rlon_flat[:5]: {rlon_flat[:5]}")
    np.testing.assert_array_almost_equal(rlon_flat, expected_rlon.ravel())
    np.testing.assert_array_almost_equal(rlat_flat, expected_rlat.ravel())


def test_rotated_rlongrid_shape():
    print("\n--- test_rotated_rlongrid_shape ---")
    g = make_rotated(nx=5, ny=4)
    rlong = g.rlongrid()
    print(f"  rlongrid shape: {rlong.shape}  (should be (4,5))")
    assert rlong.shape == (4, 5)


def test_rotated_rlatgrid_shape():
    print("\n--- test_rotated_rlatgrid_shape ---")
    g = make_rotated(nx=5, ny=4)
    rlatg = g.rlatgrid()
    print(f"  rlatgrid shape: {rlatg.shape}  (should be (4,5))")
    assert rlatg.shape == (4, 5)


def test_rlongrid_returns_none_for_spherical():
    g = make_spherical()
    assert g.rlongrid() is None


def test_rlatgrid_returns_none_for_cartesian():
    g = make_cartesian()
    assert g.rlatgrid() is None


# ------------------------------------------------------------------
# lonlat on rotated grid
# ------------------------------------------------------------------


def test_rotated_lonlat_shape():
    print("\n--- test_rotated_lonlat_shape ---")
    g = make_rotated(nx=5, ny=4)
    lon, lat = g.lonlat()
    print(f"  lon shape: {lon.shape}  (should be 20,)")
    print(f"  lat shape: {lat.shape}  (should be 20,)")
    assert lon.shape == (20,)
    assert lat.shape == (20,)


def test_rotated_lonlat_geographic_range():
    print("\n--- test_rotated_lonlat_geographic_range ---")
    g = make_rotated(nx=5, ny=4)
    lon, lat = g.lonlat()
    print(f"  lon range: [{lon.min():.3f}, {lon.max():.3f}]")
    print(f"  lat range: [{lat.min():.3f}, {lat.max():.3f}]")
    assert np.all(lon >= -180) and np.all(lon <= 180)
    assert np.all(lat >= -90) and np.all(lat <= 90)


def test_rotated_lonlat_strict_returns_none_none():
    print("\n--- test_rotated_lonlat_strict_returns_none_none ---")
    g = make_rotated()
    lon, lat = g.lonlat(strict=True)
    print(f"  lonlat(strict=True): {lon}, {lat}")
    assert lon is None
    assert lat is None


def test_rotated_lonlat_native_returns_rlonlat():
    print("\n--- test_rotated_lonlat_native_returns_rlonlat ---")
    g = make_rotated(nx=5, ny=4)
    lon, lat = g.lonlat(native=True)
    rlon, rlat = g.rlonlat()
    print(f"  lonlat(native=True) lon[:5]: {lon[:5]}")
    print(f"  rlonlat rlon[:5]:           {rlon[:5]}")
    np.testing.assert_array_almost_equal(lon, rlon)
    np.testing.assert_array_almost_equal(lat, rlat)


# ------------------------------------------------------------------
# xy on rotated grid
# ------------------------------------------------------------------


def test_rotated_xy_raises():
    print("\n--- test_rotated_xy_raises ---")
    g = make_rotated()
    with pytest.raises(ValueError, match="rlonlat"):
        g.xy()


def test_rotated_xy_strict_returns_none_none():
    print("\n--- test_rotated_xy_strict_returns_none_none ---")
    g = make_rotated()
    x, y = g.xy(strict=True)
    print(f"  xy(strict=True): {x}, {y}")
    assert x is None
    assert y is None


# ------------------------------------------------------------------
# round-trip: rlon/rlat -> lon/lat -> rlon/rlat
# ------------------------------------------------------------------


def test_rotated_roundtrip():
    print("\n--- test_rotated_roundtrip ---")
    g = make_rotated(nx=5, ny=4)
    rlon_orig, rlat_orig = g.rlonlat()
    lon, lat = g.lonlat()
    print(f"  lon range: [{lon.min():.3f}, {lon.max():.3f}]")

    # Convert true lon/lat back to rotated via proj inverse
    rlon_back = g.proj._x(lon=lon, lat=lat, proj=ROTATED_POLE)
    rlat_back = g.proj._y(lon=lon, lat=lat, proj=ROTATED_POLE)
    print(f"  max rlon diff: {np.max(np.abs(rlon_orig - rlon_back)):.6f}")
    print(f"  max rlat diff: {np.max(np.abs(rlat_orig - rlat_back)):.6f}")
    np.testing.assert_array_almost_equal(rlon_orig, rlon_back, decimal=4)
    np.testing.assert_array_almost_equal(rlat_orig, rlat_back, decimal=4)


# ------------------------------------------------------------------
# PointSkeletonExtended rotated
# ------------------------------------------------------------------


def test_point_rotated_init():
    print("\n--- test_point_rotated_init ---")
    pts = PointSkeletonExtended(rlon=[-5.0, 0.0, 5.0], rlat=[-3.0, 0.0, 3.0])
    pts.proj.set(ROTATED_POLE, silent=True)
    print(f"  is_rotated: {pts.core.is_rotated()}")
    print(f"  nx: {pts.nx()}")
    print(pts)
    assert pts.core.is_rotated()
    assert pts.nx() == 3


def test_point_rotated_rlon_rlat():
    print("\n--- test_point_rotated_rlon_rlat ---")
    pts = PointSkeletonExtended(rlon=[-5.0, 0.0, 5.0], rlat=[-3.0, 0.0, 3.0])
    pts.proj.set(ROTATED_POLE, silent=True)
    print(f"  rlon: {pts.rlon()}")
    print(f"  rlat: {pts.rlat()}")
    np.testing.assert_array_almost_equal(pts.rlon(), np.array([-5.0, 0.0, 5.0]))
    np.testing.assert_array_almost_equal(pts.rlat(), np.array([-3.0, 0.0, 3.0]))


def test_point_rotated_lon_lat_geographic():
    print("\n--- test_point_rotated_lon_lat_geographic ---")
    pts = PointSkeletonExtended(rlon=[-5.0, 0.0, 5.0], rlat=[-3.0, 0.0, 3.0])
    pts.proj.set(ROTATED_POLE, silent=True)
    lon = pts.lon()
    lat = pts.lat()
    print(f"  lon: {lon}")
    print(f"  lat: {lat}")
    assert np.all(lon >= -180) and np.all(lon <= 180)
    assert np.all(lat >= -90) and np.all(lat <= 90)


def test_point_rotated_x_native_returns_rlon():
    print("\n--- test_point_rotated_x_native_returns_rlon ---")
    pts = PointSkeletonExtended(rlon=[-5.0, 0.0, 5.0], rlat=[-3.0, 0.0, 3.0])
    pts.proj.set(ROTATED_POLE, silent=True)
    x = pts.x(native=True)
    print(f"  x(native=True): {x}")
    np.testing.assert_array_almost_equal(x, np.array([-5.0, 0.0, 5.0]))


def test_point_rotated_rlonlat():
    print("\n--- test_point_rotated_rlonlat ---")
    pts = PointSkeletonExtended(rlon=[-5.0, 0.0, 5.0], rlat=[-3.0, 0.0, 3.0])
    pts.proj.set(ROTATED_POLE, silent=True)
    rlon, rlat = pts.rlonlat()
    print(f"  rlon: {rlon}")
    print(f"  rlat: {rlat}")
    np.testing.assert_array_almost_equal(rlon, np.array([-5.0, 0.0, 5.0]))
    np.testing.assert_array_almost_equal(rlat, np.array([-3.0, 0.0, 3.0]))


def test_point_rlonlat_returns_none_none_for_spherical():
    pts = PointSkeletonExtended(lon=[5.0, 6.0], lat=[58.0, 59.0])
    rlon, rlat = pts.rlonlat()
    assert rlon is None
    assert rlat is None


# ------------------------------------------------------------------
# parametrized native/strict matrix — all three grid types
# ------------------------------------------------------------------


@pytest.mark.parametrize(
    "grid_type,make_fn",
    [
        ("spherical", make_spherical),
        ("cartesian", make_cartesian),
        ("rotated", make_rotated),
    ],
)
def test_lon_always_returns_geographic(grid_type, make_fn):
    print(f"\n--- test_lon_always_returns_geographic [{grid_type}] ---")
    g = make_fn()
    lon = g.lon()
    print(f"  [{grid_type}] lon: {lon}")
    assert lon is not None
    assert np.all(lon >= -180) and np.all(lon <= 180)


@pytest.mark.parametrize(
    "grid_type,make_fn",
    [
        ("spherical", make_spherical),
        ("cartesian", make_cartesian),
        ("rotated", make_rotated),
    ],
)
def test_lat_always_returns_geographic(grid_type, make_fn):
    print(f"\n--- test_lat_always_returns_geographic [{grid_type}] ---")
    g = make_fn()
    lat = g.lat()
    print(f"  [{grid_type}] lat: {lat}")
    assert lat is not None
    assert np.all(lat >= -90) and np.all(lat <= 90)


@pytest.mark.parametrize(
    "grid_type,make_fn,expect_none",
    [
        ("spherical", make_spherical, False),
        ("cartesian", make_cartesian, True),
        ("rotated", make_rotated, True),
    ],
)
def test_lon_strict(grid_type, make_fn, expect_none):
    print(f"\n--- test_lon_strict [{grid_type}] ---")
    g = make_fn()
    lon = g.lon(strict=True)
    print(f"  [{grid_type}] lon(strict=True): {lon}")
    assert (lon is None) == expect_none


@pytest.mark.parametrize(
    "grid_type,make_fn,expect_none",
    [
        ("spherical", make_spherical, True),
        ("cartesian", make_cartesian, False),
        ("rotated", make_rotated, True),
    ],
)
def test_x_strict(grid_type, make_fn, expect_none):
    print(f"\n--- test_x_strict [{grid_type}] ---")
    g = make_fn()
    x = g.x(strict=True)
    print(f"  [{grid_type}] x(strict=True): {x}")
    assert (x is None) == expect_none


@pytest.mark.parametrize(
    "grid_type,make_fn,expect_none",
    [
        ("spherical", make_spherical, True),
        ("cartesian", make_cartesian, True),
        ("rotated", make_rotated, False),
    ],
)
def test_rlon_none_unless_rotated(grid_type, make_fn, expect_none):
    print(f"\n--- test_rlon_none_unless_rotated [{grid_type}] ---")
    g = make_fn()
    rlon = g.rlon()
    print(f"  [{grid_type}] rlon(): {rlon}")
    assert (rlon is None) == expect_none


@pytest.mark.parametrize(
    "grid_type,make_fn,expect_raises",
    [
        ("spherical", make_spherical, False),
        ("cartesian", make_cartesian, False),
        ("rotated", make_rotated, True),
    ],
)
def test_xy_raises_only_for_rotated(grid_type, make_fn, expect_raises):
    print(f"\n--- test_xy_raises_only_for_rotated [{grid_type}] ---")
    g = make_fn()
    if expect_raises:
        with pytest.raises(ValueError):
            g.xy(proj=UTM33N)
    else:
        x, y = g.xy(proj=UTM33N)
        print(f"  [{grid_type}] xy shapes: {x.shape}, {y.shape}")
        print(f"x: {x}")
        assert x is not None
