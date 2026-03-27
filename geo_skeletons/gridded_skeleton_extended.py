from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from .skeleton_extended import SkeletonExtended
from .point_skeleton_extended import PointSkeletonExtended
from . import distance_funcs
from .managers.coordinate_manager import CoordinateManager
from .managers.dask_manager import DaskManager
from .managers.metadata_manager import MetaDataManager
from .variables import Coordinate, DataVar
import geo_parameters as gp
from typing import Optional, Union
from .dask_computations import undask_me
from pyproj import CRS, Proj
import warnings

lon_var = Coordinate(name="lon", meta=gp.grid.Lon, coord_group="spatial")
lat_var = Coordinate(name="lat", meta=gp.grid.Lat, coord_group="spatial")
rlon_var = Coordinate(name="rlon", meta=gp.grid.Rlon, coord_group="spatial")
rlat_var = Coordinate(name="rlat", meta=gp.grid.Rlat, coord_group="spatial")
x_var = Coordinate(name="x", meta=gp.grid.X, coord_group="spatial")
y_var = Coordinate(name="y", meta=gp.grid.Y, coord_group="spatial")

INITIAL_CARTESIAN_COORDS = [y_var, x_var]
INITIAL_SPHERICAL_COORDS = [lat_var, lon_var]
INITIAL_ROTATED_COORDS = [rlat_var, rlon_var]

INITIAL_VARS = []


class GriddedSkeletonExtended(SkeletonExtended):
    """Gives a gridded structure to the Skeleton.

    In practise this means that:

    1) Grid coordinates are defined as x,y / lon,lat/ rlon,rlat.
    2) Methods x(), y() / lon(), lat() will return the vectors defining the grid.
    3) Methods xy() / lonlat() will return a list of all points of the grid
    (i.e. raveled meshgrid).
    """

    meta = MetaDataManager(ds_manager=None)
    core = CoordinateManager(
        INITIAL_CARTESIAN_COORDS, INITIAL_VARS, metadata_manager=meta
    )

    @classmethod
    def from_skeleton(
        cls,
        skeleton: SkeletonExtended,
        mask: Optional[np.ndarray] = None,
    ) -> GriddedSkeletonExtended:
        """Creates a new GriddedSkeleton containing only points from another GriddedSkeleton.

        Subgrid can be selected by a boolean mask. No data is transferred"""
        if not skeleton.is_gridded():
            raise TypeError(
                "Can't create a GriddedSkeleton from a non-gridded data structure!"
            )

        if mask is None:
            mask = np.full(skeleton.size("spatial"), True)
        mask = undask_me(mask)

        lon, lat = skeleton.lon(strict=True, mask=mask), skeleton.lat(
            strict=True, mask=mask
        )
        x, y = skeleton.x(strict=True, mask=mask), skeleton.y(strict=True, mask=mask)
        rlon, rlat = skeleton.rlon(strict=True, mask=mask), skeleton.rlat(
            strict=True, mask=mask
        )

        new_skeleton = cls(
            lon=lon, lat=lat, x=x, y=y, rlon=rlon, rlat=rlat, name=skeleton.name
        )
        new_skeleton.proj.set(skeleton.proj.projection(), silent=True)

        return new_skeleton

    @staticmethod
    def is_gridded() -> bool:
        return True

    @staticmethod
    def _initial_coords(spherical: bool = False, rotated: bool = False):
        if rotated:
            return INITIAL_ROTATED_COORDS
        if spherical:
            return INITIAL_SPHERICAL_COORDS
        return INITIAL_CARTESIAN_COORDS

    @staticmethod
    def _initial_vars(spherical: bool = False, rotated: bool = False) -> list[DataVar]:
        """Initial variables used with GriddedSkeletons. Additional variables
        can be added by decorator @add_datavar.
        """
        return INITIAL_VARS

    def xgrid(
        self, native: bool = False, strict: bool = False, normalize: bool = False
    ) -> np.ndarray:
        """Gives a meshgrid of projected x-values.

        strict = True gives 'None' if Skeleton is spherical
        native = True gives longitude values if Skeleton is spherical"""
        if not self.core.is_cartesian() and strict:
            return None
        x, _ = self.xy(native=native, normalize=normalize)
        return np.reshape(x, self.size("spatial"))

    def ygrid(
        self, native: bool = False, strict: bool = False, normalize: bool = False
    ) -> np.ndarray:
        """Gives a meshgrid of projected y-values.

        strict = True gives 'None' if Skeleton is spherical
        native = True gives longitude values if Skeleton is spherical"""
        if not self.core.is_cartesian() and strict:
            return None
        _, y = self.xy(native=native, normalize=normalize)
        return np.reshape(y, self.size("spatial"))

    def longrid(self, native: bool = False, strict: bool = False) -> np.ndarray:
        """Gives a meshgrid of longitude values. 'None' for cartesian grids that have no projection.

        strict = True gives 'None' if Skeleton is cartesian
        native = True gives projected x-values if Skeleton is cartesian"""
        if self.core.is_cartesian() and strict:
            return None
        lon, _ = self.lonlat(native=native)
        if lon is None:  # Might happen if projection is not set
            return None
        return np.reshape(lon, self.size("spatial"))

    def latgrid(self, native: bool = False, strict: bool = False) -> np.ndarray:
        """Gives a meshgrid of latitude values. 'None' for cartesian grids that have no projection.

        strict = True gives 'None' if Skeleton is cartesian
        native = True gives projected y-values if Skeleton is cartesian"""
        if self.core.is_cartesian() and strict:
            return None
        _, lat = self.lonlat(native=native)

        if lat is None:  # Might happen if projection is not set
            return None
        return np.reshape(lat, self.size("spatial"))

    def rlongrid(self, strict: bool = False) -> np.ndarray:
        """Meshgrid of rotated longitudes. None if grid is not rotated."""
        if not self.core.is_rotated():
            return None
        rlon, _ = self.rlonlat()
        return np.reshape(rlon, self.size("spatial"))

    def rlatgrid(self, strict: bool = False) -> np.ndarray:
        """Meshgrid of rotated latitudes. None if grid is not rotated."""
        if not self.core.is_rotated():
            return None
        _, rlat = self.rlonlat()
        return np.reshape(rlat, self.size("spatial"))

    def x(
        self,
        native: bool = False,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
        normalize: bool = False,
        proj: CRS = None,
        **kwargs,
    ) -> np.ndarray:
        """Returns the cartesian x-coordinate.

        If the grid is spherical, a conversion to projected coordinates is made .

        strict = True gives 'None' if Skeleton is spherical
        native = True gives longitude values if Skeleton is spherical

        Give 'projection' to get cartesian coordinates in specific projection. Otherwise defaults to the one set for the grid.
        """

        mask = self._check_mask_right_shape(mask, self.core.x_str, **kwargs)
        vec_mask = np.any(mask, axis=0)
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
            x = self._ds_manager.get("x", **kwargs).values.copy()[vec_mask]

        else:

            ### This will return a 2D meshgrid !
            lon, lat = self.lon(mask=mask, **kwargs), self.lat(mask=mask, **kwargs)
            x = self.proj._x(lon=lon, lat=lat, proj=proj)

        if normalize:
            x = x - min(x)
        return x

    def y(
        self,
        native: bool = False,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
        normalize: bool = False,
        proj: Union[Proj, CRS] = None,
        **kwargs,
    ) -> np.ndarray:
        """Returns the cartesian y-coordinate.

        If the grid is spherical, a conversion to projected coordinates is made.

        strict = True gives 'None' if Skeleton is spherical
        native = True gives latitude values if Skeleton is spherical

        Give 'projection' to get cartesian coordinates in specific projection. Otherwise defaults to the one set for the grid.
        """

        mask = self._check_mask_right_shape(mask, self.core.y_str, **kwargs)
        vec_mask = np.any(mask, axis=1)
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

        if self.core.is_cartesian() and (
            self.proj.projection() == proj or proj is None
        ):
            y = self._ds_manager.get("y", **kwargs).values.copy()[vec_mask]
        else:
            ### This will return a 2D meshgrid !
            lon, lat = self.lon(mask=mask, **kwargs), self.lat(mask=mask, **kwargs)
            y = self.proj._y(lon=lon, lat=lat, proj=proj)

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

        If the grid is cartesian, a conversion from projected coordinates is made.
        If the grid is rotated, a conversion from rotated coordinates is made.

        strict = True gives 'None' if Skeleton is cartesian
        native = True gives proj x-values if Skeleton is cartesian
        """

        mask = self._check_mask_right_shape(mask, self.core.x_str, **kwargs)
        vec_mask = np.any(mask, axis=0)
        if native and strict:
            raise ValueError("Can't set both 'native' and 'strict' to True!")

        if self.ds() is None:
            return None

        if self.core.is_spherical():
            return self._ds_manager.get("lon", **kwargs).values.copy()[vec_mask]

        if not self.core.is_spherical() and strict:
            return None

        if self.core.is_cartesian():
            if native:
                return self.x(mask=mask, proj=proj, **kwargs)
            else:
                ### This will return a 2D meshgrid !
                x, y = self.x(mask=mask, proj=proj, **kwargs), self.y(
                    mask=mask, proj=proj, **kwargs
                )

                return self.proj._lon(x=x, y=y, proj=proj)

        if self.core.is_rotated():
            if native:
                warnings.warn(
                    "Using rotated longitude to represent lon coordinate. Deprecated"
                )
                return self.rlon(mask=mask, **kwargs)
            else:
                ### This will return a 2D meshgrid !
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

        If the grid is cartesian, a conversion from projected coordinates is made.
        If the grid is rotated, a conversion from rotated coordinates is made.

        strict = True gives 'None' if Skeleton is cartesian
        native = True gives projected y-values if Skeleton is cartesian
        """

        mask = self._check_mask_right_shape(mask, self.core.y_str, **kwargs)
        vec_mask = np.any(mask, axis=1)
        if native and strict:
            raise ValueError("Can't set both 'native' and 'strict' to True!")

        if self.ds() is None:
            return None

        if self.core.is_spherical():
            return self._ds_manager.get("lat", **kwargs).values.copy()[vec_mask]

        if not self.core.is_spherical() and strict:
            return None

        if self.core.is_cartesian():
            if native:
                return self.y(
                    mask=mask, proj=proj, **kwargs
                )  # TODO should maybe mask here
            else:
                ### This will return a 2D meshgrid !
                x, y = self.x(mask=mask, proj=proj, **kwargs), self.y(
                    mask=mask, proj=proj, **kwargs
                )

                return self.proj._lat(x=x, y=y, proj=proj)

        if self.core.is_rotated():

            if native:
                warnings.warn(
                    "Using rotated latitude to represent lat coordinate. Deprecated"
                )
                return self.rlat(mask=mask, **kwargs)
            else:
                ### This will return a 2D meshgrid !
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
        mask = self._check_mask_right_shape(mask, self.core.x_str, **kwargs)
        vec_mask = np.any(mask, axis=0)
        return self._ds_manager.get("rlon", **kwargs).values.copy()[vec_mask]

    def rlat(self, mask: Optional[np.ndarray] = None, **kwargs) -> np.ndarray:
        """Returns the native rotated latitude vector. None if grid is not rotated."""
        if not self.core.is_rotated():
            return None
        mask = self._check_mask_right_shape(mask, self.core.y_str, **kwargs)
        vec_mask = np.any(mask, axis=1)
        return self._ds_manager.get("rlat", **kwargs).values.copy()[vec_mask]

    def xy(
        self,
        native: bool = False,
        strict: bool = False,
        mask: Optional[np.ndarray] = None,
        proj: tuple[int, str] = None,
        normalize: bool = False,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple of projected x- and y-coordinates of all points.

        strict = True gives '(None, None)' if Skeleton is spherical
        native = True gives longitude,latitude-values if Skeleton is spherical

        Give 'proj' to get cartesian coordinates in specific projection. Otherwise defaults to the one set for the grid.

        mask is a boolean array (default True for all points)
        """

        if native and strict:
            raise ValueError("Can't set both 'native' and 'strict' to True!")
        if not self.core.is_cartesian() and strict:
            return None, None
        if self.core.is_rotated() and not strict and not native:
            raise ValueError(
                "xy() is not defined for rotated pole grids — "
                "use lonlat() for geographic coordinates or rlonlat() for native rotated coordinates."
            )
        if mask is None:
            mask = np.full(super().size("spatial", **kwargs), True)
        num_of_elements = (
            self.shape(self.core.x_str)[0] * self.shape(self.core.y_str)[0]
        )
        if mask.ravel().shape[0] != num_of_elements:
            raise ValueError(
                f"Skeleton has {num_of_elements} elements but mask has shape {mask.shape}, not ({num_of_elements},)!"
            )
        mask = mask.ravel()
        x, y = self._native_xy(proj=proj, normalize=normalize, **kwargs)
        if self.core.is_cartesian() or native:
            return x[mask], y[mask]

        # Only convert if skeleton is not Cartesian and native output is not requested
        points = PointSkeletonExtended(lon=x, lat=y)
        points.proj.set(proj or self.proj.projection(), silent=True)

        return points.xy(mask=mask, normalize=normalize, proj=proj)

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

        mask is a boolean array (default True for all points)
        """

        if native and strict:
            raise ValueError("Can't set both 'native' and 'strict' to True!")

        if not self.core.is_spherical() and strict:
            return None, None

        if mask is None:
            mask = np.full(super().size("spatial", **kwargs), True)

        num_of_elements = (
            self.shape(self.core.x_str)[0] * self.shape(self.core.y_str)[0]
        )
        if mask.ravel().shape[0] != num_of_elements:
            raise ValueError(
                f"Skeleton has {num_of_elements} elements but mask has shape {mask.shape}, not ({num_of_elements},)!"
            )
        mask = mask.ravel()
        x, y = self._native_xy(proj=proj, **kwargs)  # can be x,y / lon,lat / rlon,rlat

        if self.core.is_spherical() or native:
            return x[mask], y[mask]

        # Only convert if skeleton is Cartesian and native output is not requested
        if self.core.is_cartesian():
            points = PointSkeletonExtended(x=x, y=y)
        elif self.core.is_rotated():
            points = PointSkeletonExtended(rlon=x, rlat=y)
        points.proj.set(self.proj.projection(), silent=True)
        return points.lonlat(mask=mask)

    def rlonlat(
        self,
        mask: Optional[np.ndarray] = None,
        **kwargs,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns (rlon, rlat) — all points of the native rotated mesh.
        Returns (None, None) if grid is not rotated."""
        if not self.core.is_rotated():
            return None, None
        if mask is None:
            mask = np.full(super().size("spatial", **kwargs), True)
        mask = mask.ravel()
        rlon, rlat = np.meshgrid(self.rlon(**kwargs), self.rlat(**kwargs))
        return rlon.ravel()[mask], rlat.ravel()[mask]

    def _native_xy(
        self, proj: Optional[tuple[int, str]] = None, normalize: bool = False, **kwargs
    ) -> tuple[np.ndarray, np.ndarray]:
        """Returns a tuple of native x and y of all points."""

        if self.core.is_rotated():
            x, y = np.meshgrid(self.rlon(**kwargs), self.rlat(**kwargs))
        else:
            x, y = np.meshgrid(
                self.x(native=True, proj=proj, normalize=normalize, **kwargs),
                self.y(native=True, proj=proj, normalize=normalize, **kwargs),
            )
        return x.ravel(), y.ravel()

    def set_spacing(
        self,
        dlon: float = 0.0,
        dlat: float = 0.0,
        dx: float = 0.0,
        dy: float = 0.0,
        dm: float = 0.0,
        dnmi: float = 0.0,
        drlon=0.0,
        drlat=0.0,  # new
        nx: int = 0,
        ny: int = 0,
        floating_edge: bool = False,
    ) -> None:
        """Defines longitude and latitude vectors based on desired spacing.

        Options (priority in this order)
        nx, ny [grid points]:   Grid resolution is set to have nx points in
                                longitude and ny points in latitude direction.

        dlon, dlat [deg]:       Grid spacing is set as close to the given resolution
                                as possible (edges are fixed).

        dm [m]:                 Grid spacing is set close to dm metres.

        dnmi [nmi]:            Grid spacing is set close to dnmi nautical miles.

        dx, dy [m]:             Grid spacing is set as close as dx and dy metres as
                                possible.

        Set floating_edge=True to force exact dlon, dlat
        and instead possibly move lon_max, lat_max slightly
        to make it work (only compatibel with native coordinates).

        """

        def determine_nx(
            x_type: str, nx, dx, dm, dlon, drlon, dnmi
        ) -> tuple[int, float]:
            """Determines how many points is needed to get the desired resolution in one dimension"""
            if x_type == "x":
                lon_type = "lon"
                rlon_type = "rlon"
            else:
                lon_type = "lat"
                rlat_type = "rlat"

            x_end = self.edges(x_type, native=True)[1]

            if nx:
                return int(nx), x_end

            if dnmi:
                if self.core.is_cartesian():
                    dm = dnmi * 1850
                else:
                    dlat = dnmi / 60
                    x_km = distance_funcs.lon_in_km(np.median(self.lat()))
                    y_km = distance_funcs.lat_in_km(np.median(self.lat()))
                    if x_type == "x":
                        dlon = dlat * (y_km / x_km)
                    else:
                        dlon = dlat

            if dlon:
                nx = (
                    np.round((self.edges(lon_type)[1] - self.edges(lon_type)[0]) / dlon)
                    + 1
                )
                if floating_edge:
                    if self.core.is_cartesian():
                        raise ValueError(
                            "Grid is cartesian, so cant set exact dlon/dlat using floating_edge!"
                        )
                    x_end = self.edges(lon_type)[0] + (nx - 1) * dlon
                return int(nx), x_end
            if drlon:
                nx = (
                    np.round(
                        (self.edges(rlon_type)[1] - self.edges(rlon_type)[0]) / drlon
                    )
                    + 1
                )
                if floating_edge:
                    if self.core.is_cartesian():
                        raise ValueError(
                            "Grid is cartesian, so cant set exact dlon/dlat using floating_edge!"
                        )
                    x_end = self.edges(rlon_type)[0] + (nx - 1) * drlon
                return int(nx), x_end
            if dm:
                dx = dm

            if dx:
                nx = np.round((self.extent(x_type) / dx)) + 1
                if floating_edge:
                    if not self.core.is_cartesian():
                        raise ValueError(
                            "Grid is spherical, so cant set exact dx/dy using floating_edge!"
                        )
                    x_end = self.edges(x_type)[0] + (nx - 1) * dx
                return int(nx), x_end

            # Nothing given
            if self.core.is_rotated():
                native_vec = self.rlon() if x_type == "x" else self.rlat()
            else:
                native_vec = (
                    self.x(native=True) if x_type == "x" else self.y(native=True)
                )
            return len(native_vec), x_end

        nx, native_x_end = determine_nx("x", nx, dx, dm, dlon, drlon, dnmi)
        ny, native_y_end = determine_nx("y", ny, dy, dm, dlat, drlat, dnmi)

        # Unique to not get [0,0,0] etc. arrays if nx=1
        x_native = np.unique(np.linspace(self.x(native=True)[0], native_x_end, nx))
        y_native = np.unique(np.linspace(self.y(native=True)[0], native_y_end, ny))

        if self.core.is_cartesian():
            x = x_native
            y = y_native
            lon = None
            lat = None
            rlon = None
            rlat = None
        elif self.core.is_spherical():
            x = None
            y = None
            lon = x_native
            lat = y_native
            rlon = None
            rlat = None
        elif self.core.is_rotated():
            x = None
            y = None
            lon = None
            lat = None
            rlon = x_native
            rlat = y_native
        else:
            raise ValueError("something is wrong ...")

        old_metadata = self.meta._metadata
        self._init_structure(x, y, lon, lat, rlon, rlat)
        self.meta.set_by_dict(old_metadata)

    def _check_mask_right_shape(
        self, mask: np.ndarray, coord: str, **kwargs
    ) -> np.array:
        """Checks that the given mask is same shape as the skeleton.
        Creates a full True maks if mask is None"""
        if mask is None:
            return np.full(self.size("spatial", **kwargs), True)

        mask = np.array(mask)

        if mask.shape != self.size("spatial", **kwargs) and mask.shape != self.shape(
            coord
        ):
            raise ValueError(
                f"Skeleton has shape {self.size('spatial',**kwargs)} and {coord} has shape {self.shape(coord)} but mask is shape {mask.shape}"
            )
        return mask
