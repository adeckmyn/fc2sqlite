import pandas
import numpy as np
from pyproj import Proj

from .logger import logger


def get_grid_limits(geo):
    """Find bounding box (lat/lon) of a grid."""
    lons, lats = get_grid_boundary(geo)

    minlat = np.floor(np.min(lats)) - 1
    minlon = np.floor(np.min(lons)) - 1
    maxlat = np.ceil(np.max(lats)) + 1
    maxlon = np.ceil(np.max(lons)) + 1

    minlat = np.max([minlat, -90])
    minlon = np.max([minlon, -180])
    maxlat = np.min([maxlat, 90])
    maxlon = np.min([maxlon, 180])

    return minlon, maxlon, minlat, maxlat


def proj4_to_string(proj4):
    """Transfrom a proj4 from dictionary to string.

    Args:
        proj4: a dictionary

    Returns:
        a single string
    """
    result = " ".join([f"+{p}={proj4[p]}" for p in proj4])
    return result



def points_restrict(geo, plist):
    """Restrict the station list to points inside the current domain.

    Args:
        gid: grib handle
        plist: A pandas table with at least columns "lon" and "lat"

    Returns:
        reduced station list that contains only stations inside the domain
    """
    # 1. Get the bounding box lat/lon values
    #    that is a fast way to eliminate most outside points

    minlon, maxlon, minlat, maxlat = get_grid_limits(geo)
    nlon = geo["nlon"]
    nlat = geo["nlat"]

    # reduce the table to the bounding box
    # Make a copy! The original retains old row numbers and becomes hard to manage.
    # FIXME: for a global grid, this will delete points that are
    #        "between" the max and min longitude
    #        e.g. if the grid is [0,...,359.95], what about 359.99?
    #        the 0 meridian is pretty important, so we need to fix this
    # NOTE: minlon, maxlon are always according to [-180,180],
    #       even if the grid itself is [0,360] !!!
    if gridinfo["wrap_x"]:
        p1 = plist[(plist["lat"] >= minlat) & (plist["lat"] <= maxlat)].copy()
    else:
        p1 = plist[
            (plist["lat"] >= minlat)
            & (plist["lat"] <= maxlat)
            & (plist["lon"] >= minlon)
            & (plist["lon"] <= maxlon)
        ].copy()

    # 2. Now use grid index (requires projection)
    #    to decide which stations are inside the grid
    #    NOTE: we could skip step 1 and project all stations...

    lon = p1["lon"].tolist()
    lat = p1["lat"].tolist()

    i, j = get_gridindex(lon, lat, geo)
    if gridinfo["wrap_x"]:
        # if x wraps around the globe, don't restrict at all
        # BUT: make sure to take this into account when interpolating!
        # NOTE: Amundsen-Scott SP base has j==0 !
        # With current code, i, j == 0 are OK, but == nlat|nlon-1 needs work
        p2 = p1[(j >= 0) & (j < nlat - 1)].copy()
    else:
        p2 = p1[(i >= 0) & (i < nlon - 1) & (j >= 0) & (j < nlat - 1)].copy()
    return p2


def get_grid_points(geo):
    """Get all lat/lon co-ordinates of the grid points.

    Args:
        gid: GRIB handle
    Returns:
        Numpy arrays with all lat/lon values.
    """
    proj = Proj(geo['proj4'])
    nlon = int(geo["nlon"])
    nlat = int(geo["nlat"])
    dx = geo["dx"]
    dy = geo["dy"]

    x0, y0 = proj(geo["lon0"], geo["lat0"])

    xxx = np.empty(nlon)
    yyy = np.empty(nlat)
    for i in range(nlon):
        xxx[i] = x0 + (float(i) * dx)
    for j in range(nlat):
        yyy[j] = y0 + (float(j) * dy)

    x_v, y_v = np.meshgrid(xxx, yyy)
    lons, lats = proj(x_v, y_v, inverse=True)
    # NOTE: the inverse projection of a "wrapped" latlong
    #       turns out to be [-180,180]
    return lons, lats


def get_grid_boundary(geo):
    """Get lat/lon co-ordinates of the grid boundary points.

    Args:
        gid: GRIB handle
    Returns:
        Numpy arrays with lat/lon values.
    """
    proj = Proj(geo['proj4'])
    nlon = int(geo["nlon"])
    nlat = int(geo["nlat"])
    dx = geo["dx"]
    dy = geo["dy"]
    # get SW corner
    x0, y0 = proj(geo["lon0"], geo["lat0"])
    xxx = np.fromiter((x0 + i * dx for i in range(nlon)), float)
    yyy = np.fromiter((y0 + i * dy for i in range(nlat)), float)
    x_v = np.concatenate(
        (
            xxx,
            np.full(nlat - 2, xxx[nlon - 1]),
            xxx[::-1],
            np.full(nlat - 2, xxx[0]),
        )
    )
    y_v = np.concatenate(
        (
            np.full(nlon - 1, yyy[0]),
            yyy,
            np.full(nlon - 2, yyy[nlat - 1]),
            yyy[:0:-1],
        )
    )

    lons, lats = proj(x_v, y_v, inverse=True)
    # NOTE: the inverse projection of a "wrapped" latlong
    #       turns out to be [-180,180]
    return lons, lats


def get_gridindex(lon, lat, geo):
    """Convert lat/lon values to (zero-offset) grid indices.

    The SW corner has index (0,0). Points are projected on the grid
    and a (non-integer) index is calculated.

    Args:
        lon: single value or list of longitude values
        lat: single value of list of latitude values
        gid: grib handle

    Returns:
        two vectors with (non-integer) index values.
    """
    proj = Proj(geo['proj4'])
    x0, y0 = proj(geo["lon0"], geo["lat0"])
    dx = geo["dx"]
    dy = geo["dy"]

    x, y = proj(np.array(lon), np.array(lat))
    i = (x - x0) / dx
    j = (y - y0) / dy
    return i, j


def train_weights(station_list, geo, lsm=False):
    """Train weights for bilinear and nearest neighbour interpolation.

    We train two kinds of interpolation at once.
    NOTE: Land/Sea Mask is not yet implemented.

    Args:
        station_list: pandas table
        gid: grib handle
        lsm: use Land/Sea mask (ignored for now)

    Returns:
        interpolation weights (nearest neighbour & bilinear)
    """
    # TODO: land/sea mask for T2m...
    if lsm:
        logger.warning("SQLITE: ignoring land/sea mask!")
    # usually, station_list is a pandas table
    # but why not also allow dicts with 'lat' and 'lon' lists
    if isinstance(station_list, dict):
        lat = np.array(station_list["lat"])
        lon = np.array(station_list["lon"])
    else:
        lat = np.array(station_list["lat"].tolist())
        lon = np.array(station_list["lon"].tolist())

    nstations = len(lon)
    i, j = get_gridindex(lon, lat, geo)

    # eccodes python interface only gives us distances
    # so we need to do some math for bilinear weights
    nearestweights = []
    bilinweights = []
    # assuming the 4 closest points are exactly what we need (OK if dx=dy)
    # NOTE: this assumes dx == dy !!!
    # otherwise, you have to check whether 2nd point is along X or Y axis from first
    # probably easy, just look at the index
    ic = np.round(i).astype("i4")
    jc = np.round(j).astype("i4")
    # FIXME: in rare cases, when i/j are exactly integer, floor==ceil
    # this happens e.g. with Amundsen-Scott South Pole base, j==0.
    # In such cases we should interpolate between only 2 points.
    i0 = np.floor(i).astype("i4")
    i1 = i0 + 1  # np.ceil(i)
    # In the very exceptional case that i0==nlon-1 (*exactly* on the outside border)
    # we may not use i0+1, but since the weights will be zero anyway, we can change to
    # any other value...
    # if i0 == nlon - 1:
    #    if i != i0:
    #        # The points were not correctly constrained to domain!
    j0 = np.floor(j).astype("i4")
    j1 = j0 + 1  # np.ceil(j)
    di = i - i0
    dj = j - j0
    w1 = (1 - di) * (1 - dj)
    w2 = (1 - di) * dj
    w3 = di * (1 - dj)
    w4 = di * dj
    if geo["wrap_x"]:
        logger.info("The domain wraps around the globe.")
        ic[ic == geo["nlon"]] = 0
        ic[ic == -1] = geo["nlon"]
        i1[i1 == geo["nlon"]] = 0
        i0[i0 == -1] = geo["nlon"]

    for pp in range(nstations):
        nearestweights.append([[[ic[pp], jc[pp]], 1.0]])
        bilinweights.append(
            [
                [[i0[pp], j0[pp]], w1[pp]],
                [[i0[pp], j1[pp]], w2[pp]],
                [[i1[pp], j0[pp]], w3[pp]],
                [[i1[pp], j1[pp]], w4[pp]],
            ]
        )

    weights = {"linear": bilinweights, "nearest": nearestweights}
    return weights


def interp_from_weights(field_data, weights, method):
    """Interpolate a GRIB field to points using the given weights.

    Args:
        field_data: a data field
        weights: interpolation weights
        method: interpolation method (linear, nearest)

    Returns:
        interpolated values
    """
    # NOTE: this assumes all records in a file use the same grid representation

    interp = [sum([field_data[x[0][0], x[0][1]] * x[1] for x in w]) for w in weights[method]]
    return interp


def get_grid_values(gid):
    """Decode a GRIB2 field.

    Args:
      gid: A grib handle

    Returns:
      Numpy array of decoded field.
    """
    gkeys = [
        "Nx",
        "Ny",
        "iScansNegatively",
        "jScansPositively",
        "jPointsAreConsecutive",
        "alternativeRowScanning",
    ]
    ginfo = get_keylist(gid, gkeys, "long")
    nx = ginfo["Nx"]
    ny = ginfo["Ny"]

    # data given by column (C) or by row (Fortran)?
    order = "C" if ginfo["jPointsAreConsecutive"] else "F"
    data = eccodes.codes_get_values(gid).reshape(nx, ny, order=order)
    if ginfo["iScansNegatively"]:
        # logger.warning("Untested data ordering iScansNegatively=1")
        data[range(nx), :] = data[range(nx)[::-1], :]
    if not ginfo["jScansPositively"]:
        # logger.warning("Untested data ordering jScansPositively=0")
        data[:, range(ny)] = data[:, range(ny)[::-1]]

    return data


