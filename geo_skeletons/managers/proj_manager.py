from .metadata_manager import MetaDataManager
import numpy as np
from typing import Optional, Union
from pyproj import Proj, CRS, Transformer, transform


class ProjManager:
    def __init__(self, metadata_manager: MetaDataManager):
        self._meta: MetaDataManager = metadata_manager
        self._projection: Union[Proj, CRS] = None

    def projection(self) -> Proj:
        """Returns projection object . Returns None
        if it hasn't been set by the user in cartesian grids."""
        return self._projection

    def is_valid(self, proj: Union[Proj, CRS]):
        pass

    def is_set(self) -> bool:
        """Checks if the projection has been set"""
        return not self._zone == (None, None)

    def set(self, proj: Optional[Union[Proj, CRS]], silent: bool = False) -> None:
        """Set UTM zone and number to be used for cartesian coordinates."""

        if proj is None:
            return
        if not self.is_valid(proj):
            raise ValueError(f"{proj} is not a valid projection!")

        self._projection = (
            proj  # add a method to sanitize the proj in Pyproj proj system
        )
        self._meta.append({"projection": f"{self._projection}"})

        if not silent:
            print(f"Setting Projection {self._projection}")

    def _lat(self, x: np.ndarray, y: np.ndarray, proj: Union[Proj, CRS]) -> np.ndarray:
        """Calculates latitudes based on given x,y-coordinates and the set Projection"""
        pass

    def _lon(self, x: np.ndarray, y: np.ndarray, proj: Union[Proj, CRS]) -> np.ndarray:
        """Calculates longitudes based on given x,y-coordinates and the set Projection"""
        pass

    def _x(
        self, lon: np.ndarray, lat: np.ndarray, proj: Union[Proj, CRS]
    ) -> np.ndarray:
        """Calculates x-coordinates based on given lon,lat-coordinates and the set Projection.
        consider using median lat for utm"""
        pass

    def _y(
        self, lon: np.ndarray, lat: np.ndarray, proj: Union[Proj, CRS]
    ) -> np.ndarray:
        """Calculates y-coordinates based on given lon,lat-coordinates and the set Projection.
        consider using median lat for utm"""
        pass

    def _Transform_vector_matrix(self):  # Maybe move somewhere else
        pass
