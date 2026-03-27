from .metadata_manager import MetaDataManager
import numpy as np
from typing import Optional, Union
from pyproj import Proj, CRS, Transformer
import warnings


def is_rotated_proj(crs: CRS) -> bool:
    cf = crs.to_cf()
    proj_dict = crs.to_dict()
    return "rotated" in cf.get("grid_mapping_name", "") or "ob_tran" in proj_dict.get(
        "proj", ""
    )


class ProjManager:
    def __init__(self, metadata_manager: MetaDataManager):
        self._meta: MetaDataManager = metadata_manager
        self._projection: Optional[CRS] = None

    def projection(self) -> CRS:
        """Returns the CRS object. Returns None if not set."""
        return self._projection

    def is_valid(self, proj) -> bool:
        """A valid projection is simply a pyproj.CRS instance."""
        return isinstance(proj, CRS)

    def is_set(self) -> bool:
        """Checks if the projection has been set"""
        return self._projection is not None

    def set(self, proj: Optional[Union[str, CRS]], silent: bool = False) -> None:
        """Set the projection. Accepts a CRS object or anything CRS.from_user_input() understands."""
        if proj is None:
            return

        # Attempt to coerce strings/proj4/EPSG ints to CRS
        if not self.is_valid(proj):
            try:
                proj = CRS.from_user_input(proj)
            except Exception as e:
                raise ValueError(f"{proj} is not a valid projection: {e}")
        # add a method to sanitize the proj in Pyproj proj system
        self._projection = proj
        self._meta.append({"projection": f"{proj.to_wkt}"})

        if not silent:
            print(f"Setting Projection {self._projection.name}")

    def _transformer_to_lonlat(self, proj: Optional[CRS] = None) -> Transformer:
        """Build a Transformer from the given (or set) CRS to WGS84."""
        crs = proj or self._projection
        if crs is None:
            raise ValueError("No projection set. Call proj.set() first.")
        return Transformer.from_crs(crs, CRS.from_epsg(4326), always_xy=True)

    def _transformer_from_lonlat(self, proj: Optional[CRS] = None) -> Transformer:
        """Build a Transformer from WGS84 to the given (or set) CRS."""
        crs = proj or self._projection
        if crs is None:
            raise ValueError("No projection set. Call proj.set() first.")
        return Transformer.from_crs(CRS.from_epsg(4326), crs, always_xy=True)

    def _resolve_shapes(self, a: np.ndarray, b: np.ndarray, a_name: str, b_name: str):
        """Return (a, b) with matching shapes, expanding to meshgrid if both are 1D.
        Raises ValueError if shapes don't match and can't be resolved."""
        a, b = np.asarray(a), np.asarray(b)
        if np.shape(a) == np.shape(b):
            return a, b
        if a.ndim == 1 and b.ndim == 1:
            warnings.warn(
                f"{a_name} {np.shape(a)} and {b_name} {np.shape(b)} shapes don't match "
                f"— expanding to meshgrid."
            )
            return np.meshgrid(a, b)
        raise ValueError(
            f"{a_name} {np.shape(a)} and {b_name} {np.shape(b)} shapes don't match "
            f"and cannot be resolved (inputs must be 1D for meshgrid expansion)."
        )

    def _lon(
        self, x: np.ndarray, y: np.ndarray, proj: Optional[CRS] = None
    ) -> np.ndarray:
        x, y = self._resolve_shapes(x, y, "x", "y")
        lon, _ = self._transformer_to_lonlat(proj).transform(x, y)
        return lon

    def _lat(
        self, x: np.ndarray, y: np.ndarray, proj: Optional[CRS] = None
    ) -> np.ndarray:
        x, y = self._resolve_shapes(x, y, "x", "y")
        _, lat = self._transformer_to_lonlat(proj).transform(x, y)
        return lat

    def _x(
        self, lon: np.ndarray, lat: np.ndarray, proj: Optional[CRS] = None
    ) -> np.ndarray:
        if proj is None:
            proj = self.projection()
        if self.projection().is_projected and is_rotated_proj(proj):
            raise ValueError(
                "Reprojecting cartesian coordinates on rotated coordinates is not allowed."
            )
        lon, lat = self._resolve_shapes(lon, lat, "lon", "lat")
        x, _ = self._transformer_from_lonlat(proj).transform(lon, lat)
        return x

    def _y(
        self, lon: np.ndarray, lat: np.ndarray, proj: Optional[CRS] = None
    ) -> np.ndarray:
        if proj is None:
            proj = self.projection()
        if self.projection().is_projected and is_rotated_proj(proj):
            raise ValueError(
                "Reprojecting cartesian coordinates on rotated coordinates is not allowed."
            )
        lon, lat = self._resolve_shapes(lon, lat, "lon", "lat")
        _, y = self._transformer_from_lonlat(proj).transform(lon, lat)
        return y

    def _Transform_vector_matrix(self):  # Maybe move somewhere else
        pass
