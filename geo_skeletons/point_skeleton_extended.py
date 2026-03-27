from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .skeleton_extended import SkeletonExtended
from .managers.coordinate_manager import CoordinateManager
from .managers.metadata_manager import MetaDataManager
from .managers.dask_manager import DaskManager
from .variables import DataVar, Coordinate
import geo_parameters as gp
from typing import Optional
from .dask_computations import undask_me
import warnings

inds_coord = Coordinate(name="inds", meta=gp.grid.Inds, coord_group="spatial")
INITIAL_COORDS = [inds_coord]

lon_var = DataVar(name="lon", meta=gp.grid.Lon, coord_group="spatial", default_value=0)
lat_var = DataVar(name="lat", meta=gp.grid.Lat, coord_group="spatial", default_value=0)
rlon_var = DataVar(
    name="rlon", meta=gp.grid.Rlon, coord_group="spatial", default_value=0
)
rlat_var = DataVar(
    name="rlat", meta=gp.grid.Rlat, coord_group="spatial", default_value=0
)
x_var = DataVar(name="x", meta=gp.grid.X, coord_group="spatial", default_value=0)
y_var = DataVar(name="y", meta=gp.grid.Y, coord_group="spatial", default_value=0)
INITIAL_CARTESIAN_VARS = [x_var, y_var]  # {x: "inds", "y": "inds"}
INITIAL_SPHERICAL_VARS = [lon_var, lat_var]  # {"lat": "inds", "lon": "inds"}
INITIAL_ROTATED_VARS = [rlon_var, rlat_var]  # {"rlat": "inds", "rlon": "inds"}


class PointSkeletonExtended(SkeletonExtended):
    """Gives a unstructured structure to the Skeleton.

    In practise this means that:

    1) Grid coordinates are defined with and index (inds),
    2) x,y / lon,lat values are data variables of the index.
    3) Methods x(), y() / lon(), lat() will returns all points.
    4) Methods xy() / lonlat() are identical to e.g. (x(), y()).
    """

    meta = MetaDataManager(ds_manager=None)
    core = CoordinateManager(
        INITIAL_COORDS, INITIAL_CARTESIAN_VARS, metadata_manager=meta
    )

    @classmethod
    def from_skeleton(
        cls,
        skeleton: SkeletonExtended,
        mask: Optional[np.ndarray] = None,
    ) -> PointSkeleton:
        """Creates a new PointSkeleton containing only points from another Gridded- or PointSkeleton.

        Points can be selected by a boolean mask. No data is transferred"""

        if mask is None:
            mask = np.full(skeleton.size("spatial"), True)
        mask = undask_me(mask)
        lon, lat = skeleton.lonlat(strict=True, mask=mask)
        x, y = skeleton.xy(strict=True, mask=mask)

        new_skeleton = cls(lon=lon, lat=lat, x=x, y=y, name=skeleton.name)
        new_skeleton.proj.set(skeleton.proj.projection(), silent=True)

        return new_skeleton

    @staticmethod
    def is_gridded() -> bool:
        return False

    @staticmethod
    def _initial_coords(
        spherical: bool = False, rotated: bool = False
    ) -> list[Coordinate]:
        """Initial coordinates used with PointSkeletons. Additional coordinates
        can be added by decorators (e.g. @add_coord, @add_time).
        """
        return INITIAL_COORDS

    @staticmethod
    def _initial_vars(spherical: bool = False, rotated: bool = False) -> list[DataVar]:
        """Initial variables used with PointSkeletons. Additional variables
        can be added by decorator @add_datavar.
        """
        if spherical:
            return INITIAL_SPHERICAL_VARS
        elif rotated:
            return INITIAL_ROTATED_VARS
        else:
            return INITIAL_CARTESIAN_VARS

    def xgrid(
        self,
        native: bool = False,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
        normalize: bool = False,
    ) -> np.ndarray:
        """Gives a meshgrid of projected x-values.

        NB! Identical to Skeleton.lat() since PointSkeletons are not gridded!

        strict = True gives 'None' if Skeleton is spherical
        native = True gives longitude values if Skeleton is spherical"""
        x, _ = self.xy(native=native, strict=strict, normalize=normalize, mask=mask)
        return x

    def ygrid(
        self,
        native: bool = False,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
        normalize: bool = False,
    ) -> np.ndarray:
        """Gives a meshgrid of UTM x-values.

        NB! Identical to Skeleton.lat() since PointSkeletons are not gridded!

        strict = True gives 'None' if Skeleton is spherical
        native = True gives longitude values if Skeleton is spherical"""
        _, y = self.xy(native=native, strict=strict, normalize=normalize, mask=mask)
        return y

    def longrid(
        self,
        native: bool = False,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Gives a meshgrid of longitude values. 'None' for cartesian grids that have no UTM-zone.

        NB! Identical to Skeleton.lat() since PointSkeletons are not gridded!

        strict = True gives 'None' if Skeleton is cartesian
        native = True gives UTM x-values if Skeleton is cartesian"""
        lon, _ = self.lonlat(native=native, strict=strict, mask=mask)
        return lon

    def latgrid(
        self,
        native: bool = False,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Gives a meshgrid of latitude values. 'None' for cartesian grids that have no UTM-zone.

        NB! Identical to Skeleton.lat() since PointSkeletons are not gridded!

        strict = True gives 'None' if Skeleton is cartesian
        native = True gives UTM y-values if Skeleton is cartesian"""
        _, lat = self.lonlat(native=native, strict=strict, mask=mask)
        return lat

    def rlongrid(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns rotated longitudes. None if grid is not rotated.
        Identical to rlon() since PointSkeletons are not gridded."""
        if not self.core.is_rotated():
            return None
        return self.rlon(mask=mask)

    def rlatgrid(self, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Returns rotated latitudes. None if grid is not rotated.
        Identical to rlat() since PointSkeletons are not gridded."""
        if not self.core.is_rotated():
            return None
        return self.rlat(mask=mask)

    def x(
        self,
        native: bool = False,
        strict: bool = False,
        proj: Optional[tuple[int, str]] = None,
        mask: Optional[np.ndarray] = None,
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Returns the cartesian x-coordinate.

        strict = True gives 'None' if Skeleton is spherical
        native = True gives longitude values if Skeleton is spherical

        Give 'proj' to get cartesian coordinates in specific projection. Otherwise defaults to the one set for the grid.
        """

        mask = self._check_mask_right_shape(mask, **kwargs)
        if native and strict:
            raise ValueError("Can't set both 'native' and 'strict' to True!")

        if self.ds() is None:
            return None

        if not self.core.is_cartesian() and native:
            if self.core.is_spherical():
                return self.lon(mask=mask, proj=proj, **kwargs)
            elif self.core.is_rotated():
                return self.rlon(mask=mask, proj=proj, **kwargs)

        if not self.core.is_cartesian() and strict:
            return None

        if self.core.is_rotated():
            raise ValueError(
                "x()/y() are not defined for rotated pole grids — "
                "use rlon()/rlat() for native coordinates or lon()/lat() for true geographic coordinates."
            )

        if self.core.is_cartesian() and (
            self.proj.projection() == proj or proj is None
        ):
            x = self._ds_manager.get("x", **kwargs).values.copy()[mask]
        else:
            # TODO check if this return 2D meshgrid or 1D vec
            x = self.proj._x(
                lon=self.lon(mask=mask, **kwargs),
                lat=self.lat(mask=mask, **kwargs),
                proj=proj,
            )

        if normalize:
            x = x - min(x)

        return x

    def y(
        self,
        native: bool = False,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
        proj: Optional[tuple[int, str]] = None,
        normalize: bool = False,
        **kwargs,
    ) -> np.ndarray:
        """Returns the cartesian y-coordinate.

        strict = True gives 'None' if Skeleton is spherical
        native = True gives latitude values if Skeleton is spherical

        Give 'proj' to get cartesian coordinates in specific projection. Otherwise defaults to the one set for the grid.
        """

        mask = self._check_mask_right_shape(mask, **kwargs)

        if native and strict:
            raise ValueError("Can't set both 'native' and 'strict' to True!")
        if self.ds() is None:
            return None

        if not self.core.is_cartesian() and native:
            if self.core.is_spherical():
                return self.lat(mask=mask, proj=proj, **kwargs)
            elif self.core.is_rotated():
                return self.rlat(mask=mask, proj=proj, **kwargs)

        if not self.core.is_cartesian() and strict:
            return None
        if self.core.is_rotated():
            raise ValueError(
                "x()/y() are not defined for rotated pole grids — "
                "use rlon()/rlat() for native coordinates or lon()/lat() for true geographic coordinates."
            )
        proj = proj or self.proj.projection()

        if self.core.is_cartesian() and (self.proj.projection() == proj):
            y = self._ds_manager.get("y", **kwargs).values.copy()[mask]
        else:
            # TODO check if this return 2D meshgrid or 1D vec
            y = self.proj._y(
                lon=self.lon(mask=mask, **kwargs),
                lat=self.lat(mask=mask, **kwargs),
                proj=proj,
            )

        if normalize:
            y = y - min(y)

        return y

    def lon(
        self,
        native: bool = False,
        strict=False,
        mask: Optional[np.ndarray] = None,
        proj: Optional[tuple[int, str]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Returns the spherical lon-coordinate. 'None' for cartesian grids that have no projection.

        strict = True gives 'None' if Skeleton is cartesian
        native = True gives projected x-values if Skeleton is cartesian
        """

        mask = self._check_mask_right_shape(mask, **kwargs)

        if native and strict:
            raise ValueError("Can't set both 'native' and 'strict' to True!")

        if self.ds() is None:
            return None

        if self.core.is_spherical():
            return self._ds_manager.get("lon", **kwargs).values.copy()[mask]

        if not self.core.is_spherical() and strict:
            return None

        if self.core.is_cartesian():
            if native:
                return self.x(mask=mask, proj=proj, **kwargs)
            else:
                # TODO check if this return 2D meshgrid or 1D vec
                return self.proj._lon(
                    x=self.x(mask=mask, proj=proj, **kwargs),
                    y=self.y(mask=mask, proj=proj, **kwargs),
                    proj=proj,
                )
        if self.core.is_rotated():
            if native:
                warnings.warn(
                    "Using rotated longitude to represent lon coordinate. Deprecated"
                )
                return self.rlon(mask=mask, **kwargs)
            else:
                # TODO check if this return 2D meshgrid or 1D vec
                rlon, rlat = (
                    self.rlon(mask=mask, **kwargs),
                    self.rlat(mask=mask, **kwargs),
                )
                return self.proj._lon(x=rlon, y=rlat, proj=self.proj.projection())

        # should be in one of the cases above otherwise raise an error
        raise ValueError("something is wrong...")

    def lat(
        self,
        native: bool = False,
        strict=False,
        mask: Optional[np.ndarray] = None,
        proj: Optional[tuple[int, str]] = None,
        **kwargs,
    ) -> np.ndarray:
        """Returns the spherical lat-coordinate. 'None' for cartesian grids that have no projection.

        strict = True gives 'None' if Skeleton is cartesian
        native = True gives projected y-values if Skeleton is cartesian
        """

        mask = self._check_mask_right_shape(mask, **kwargs)

        if native and strict:
            raise ValueError("Can't set both 'native' and 'strict' to True!")

        if self.ds() is None:
            return None

        if self.core.is_spherical():
            return self._ds_manager.get("lat", **kwargs).values.copy()[mask]

        if not self.core.is_spherical() and strict:
            return None

        if self.core.is_cartesian():
            if native:
                return self.y(mask=mask, proj=proj, **kwargs)
            else:
                # TODO check if this return 2D meshgrid or 1D vec
                return self.proj._lat(
                    x=self.x(mask=mask, proj=proj, **kwargs),
                    y=self.y(mask=mask, proj=proj, **kwargs),
                    proj=proj,
                )

        if self.core.is_rotated():

            if native:
                warnings.warn(
                    "Using rotated latitude to represent lat coordinate. Deprecated"
                )
                return self.rlat(mask=mask, **kwargs)
            else:
                # TODO check if this return 2D meshgrid or 1D vec
                rlon, rlat = (
                    self.rlon(mask=mask, **kwargs),
                    self.rlat(mask=mask, **kwargs),
                )
                return self.proj._lat(x=rlon, y=rlat, proj=self.proj.projection())

        # should be in one of the cases above otherwise raise an error
        raise ValueError("something is wrong...")

    def rlon(self, mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Returns the native rotated longitude vector. None if grid is not rotated."""
        if not self.core.is_rotated():
            return None
        mask = self._check_mask_right_shape(mask, **kwargs)
        return self._ds_manager.get("rlon", **kwargs).values.copy()[mask]

    def rlat(self, mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Returns the native rotated latitude vector. None if grid is not rotated."""
        if not self.core.is_rotated():
            return None
        mask = self._check_mask_right_shape(mask, **kwargs)
        return self._ds_manager.get("rlat", **kwargs).values.copy()[mask]

    def xy(
        self,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
        proj: Optional[tuple[int, str]] = None,
        normalize: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple of projected x- and y-coordinates of all points.

        strict = True gives '(None, None)' if Skeleton is spherical
        native = True gives longitude,latitude-values if Skeleton is spherical

        Give 'proj' to get cartesian coordinates in specific projection. Otherwise defaults to the one set for the grid.

        Identical to (.x(), .y()) (with no mask)
        mask is a boolean array (default True for all points)
        """

        mask = self._check_mask_right_shape(mask, **kwargs)

        # Transforms x-y to lon-lat if necessary
        x, y = self.x(
            strict=strict, normalize=normalize, proj=proj, mask=mask, **kwargs
        ), self.y(strict=strict, normalize=normalize, proj=proj, mask=mask, **kwargs)

        if x is None:
            return None, None

        return x, y

    def lonlat(
        self,
        native: bool = False,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
        proj: Optional[tuple[int, str]] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple of longitude and latitude of all points.

        strict = True gives '(None, None)' if Skeleton is cartesian
        native = True gives projected x,y-values if Skeleton is cartesian

        Identical to (.lon(), .lat()) (with no mask)
        mask is a boolean array (default True for all points)
        """

        mask = self._check_mask_right_shape(mask)

        lon, lat = self.lon(
            native=native, strict=strict, mask=mask, proj=proj, **kwargs
        ), self.lat(native=native, strict=strict, mask=mask, proj=proj, **kwargs)

        if lon is None:
            return None, None
        return lon, lat

    def rlonlat(
        self,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (rlon, rlat) for all points.
        Returns (None, None) if grid is not rotated."""
        if not self.core.is_rotated():
            return None, None
        mask = self._check_mask_right_shape(mask, **kwargs)
        rlon = self.rlon(mask=mask, **kwargs)
        rlat = self.rlat(mask=mask, **kwargs)
        if rlon is None:
            return None, None
        return rlon, rlat

    def _check_mask_right_shape(self, mask: np.ndarray, **kwargs) -> np.array:
        """Checks that the given mask is same shape as the skeleton.
        Creates a full True maks if mask is None"""
        if mask is None:
            return np.full(self.size("spatial", **kwargs), True)
        mask = np.array(mask)

        if mask.shape != self.size("spatial", **kwargs):
            raise ValueError(
                f"Skeleton has shape {self.size('spatial',**kwargs)} but mask is shape {mask.shape}"
            )
        return mask
