#! /usr/bin/env python3
"""fc2sqlite: toolbox for extracting point data from GRIB fields."""
from .fa_tools import *
from .sqlite_tools import sqlite_name, create_table, write_to_sqlite
from .phys_functions import param_apply_function

import os
import json
import struct
from copy import deepcopy
from datetime import datetime

import pandas
import eccodes
import numpy as np
from pyproj import Proj

from .logger import logger

# some defaults
basedir = os.path.join(os.path.dirname(__file__), "data")
default_parameter_list = "param_list_default.json"
default_station_list = "station_list_default.csv"
default_sqlite_template = "{MODEL}/{YYYY}/{MM}/FCTABLE_{PP}_{YYYY}{MM}.sqlite"
# TODO: use epygram.formats.guess(filename)


def get_file_type(filename):
    """Find the format (grib or fa).

    Args:
        filename: str
    Returns:
        a string ("GRIB" or "FA") or None
    """

    with open(filename, "rb") as infile:
        infile.seek(0)
        header = infile.read(5*8)
        if header[0:4] == b'GRIB':
            return("GRIB")
        else:
            faheader = list(struct.unpack(">5Q", header))
            if faheader[1] == 16 and faheader[3] == 22:
                return("FA")
    return(None)


def read_param_list(param_file = None):
    """Read a parameter file (json).

    Args:
        param_file: name of a json file

    Returns:
        A dict containing the parsed parameter list
    """
    if param_file is None:
        param_file = default_parameter_list

    if not os.path.isfile(param_file) and os.path.dirname(param_file) == "":
        param_file = os.path.join(basedir, param_file)

    try:
        with open(param_file) as pf:
            parameter_list = json.load(pf)
            pf.close()
    except OSError:
        logger.error("Can not read parameter file %s", param_file)
        raise
    
    if "macros" in parameter_list[0]:
        # TODO: this is concept code, I'm sure it can be improved.
        logger.info("Resolving macros in parameter list.")
        macros = parameter_list.pop(0)['macros']
        for param in parameter_list:
            for mac in macros.keys():
                for key in param.keys():
                    if param[key] == mac:
                        param[key] = macros[mac]
    return parameter_list


def read_station_list(station_file = None):
    """Read a station file (csv).

    Args:
        station_file: name of a csv file

    Returns:
        A pandas table containing the parsed station list
    """
    if station_file is None:
        station_file = default_station_list

    if not os.path.isfile(station_file) and os.path.dirname(station_file) == "":
        station_file = os.path.join(basedir, station_file)

    try:
        station_list = pandas.read_csv(station_file, skipinitialspace=True)
    except OSError:
        logger.error("Can not read station file %s", station_file)
        raise
    return station_list


def get_date_info(gid):
    """Forecast time, lead time, accumulation time etc. from GRIB record.

    Args:
        gid: grib handle

    Returns:
        Forecast date/time and lead time as read from the grib record
    """
    keys = [
        "editionNumber",
        "dataDate",
        "hour",
        "minute",
        "second",
    ]
    info = get_keylist(gid, keys, "long")
    if info["editionNumber"] == 1:
        logger.info("GRIB-1 files are experimental.")
        keys2 = [
            "stepUnits",
            "endStep",
        ]
        info2 = get_keylist(gid, keys2, "long")
        info = info | info2

    else:
        # NOTE: using stepUnits as number is bugged (Nov 2024) for sub-hourly
        #       certainly in grib-2, probably also grib-1
        #       but avoid grib-1 for sub-hourly data!
        keys2 = [
            "indicatorOfUnitOfTimeRange",
            "forecastTime",
            "productDefinitionTemplateNumber",
        ]
        info2 = get_keylist(gid, keys2, "long")
        info = info | info2

        if info2["productDefinitionTemplateNumber"] == 8:
            keys3 = [
                "indicatorOfUnitForTimeRange",
                "lengthOfTimeRange",
            ]
            info3 = get_keylist(gid, keys3, "long")
            info = info | info3
    return date_from_gribinfo(info)


def date_from_gribinfo(info):
    """Interprete list of date information as given by eccodes.

    Args:
        info: a dict with all necessary grib key values

    Returns:
        forecast date (datetime object) and lead time (in seconds!)

    """
    fcdate = datetime.strptime(
        "{:8}T{:02}{:02}{:02}".format(
            info["dataDate"], info["hour"], info["minute"], info["second"]
        ),
        "%Y%m%dT%H%M%S",
    )

    if info["editionNumber"] == 1:
        tunit = info["stepUnits"]
        leadtime = info["endStep"]
    else:
        tunit = info["indicatorOfUnitOfTimeRange"]
        leadtime = info["forecastTime"]

    if tunit == 1:
        tscale = 3600.0
    elif tunit == 2:
        tscale = 60.0
    elif tunit == 13:
        tscale = 1.0
    else:
        logger.error("Unrecognised indicatorOfUnitOfTimeRange: %i", tunit)
        return None

    leadtime *= tscale

    # At this point, leadtime may be just the START of accumulation (or min/max/mean) time
    if info["editionNumber"] == 2 and info["productDefinitionTemplateNumber"] == 8:
        tunit2 = info["indicatorOfUnitForTimeRange"]  # NOTE: "For" vs "Of"
        if tunit2 == 1:
            lt2 = info["lengthOfTimeRange"] * 3600.0
        elif tunit2 == 2:
            lt2 = info["lengthOfTimeRange"] * 60.0
        elif tunit2 == 13:
            lt2 = info["lengthOfTimeRange"]
        else:
            logger.error("Unrecognised indicatorOfUnitForTimeRange: %i", tunit2)
            return None
        leadtime += lt2

    return (fcdate, leadtime)


def get_proj4(gid):
    """Read all projection details and return proj4 string.

    Args:
        gid: GRIB handle

    Returns:
        projection function corresponding to proj4 string
    """
    # NOTE: for now, we assume standard ACCORD Earth shape etc.
    # FIXME: for GRIB-2, we should retrieve the actual earth shape!
    proj4 = {"R": 6371229}
    gridtype = get_keylist(gid, ["gridType"], "string")["gridType"]
    if gridtype in ["lambert", "lambert_lam"]:
        pkeys = ["Latin1InDegrees", "Latin2InDegrees", "LoVInDegrees"]
        p4 = get_keylist(gid, pkeys, "double")

        proj4["proj"] = "lcc"
        proj4["lon_0"] = p4["LoVInDegrees"]
        proj4["lat_1"] = p4["Latin1InDegrees"]
        proj4["lat_2"] = p4["Latin2InDegrees"]
        # TODO: Northern of Southern hemisphere? projectionCentreFlag (?)

    elif gridtype == "polar_steoreographic":
        pkeys = ["LoVInDegrees"]
        p4 = get_keylist(gid, pkeys, "double")
        proj4["proj"] = "stere"
        proj4["lon_0"] = p4["LoVInDegrees"]
        proj4["lat_0"] = 90.0  # FIXME: assuming Northern hemisphere!
    elif gridtype == "regular_ll":
        pkeys = [
            "longitudeOfFirstGridPointInDegrees",
            "longitudeOfLastGridPointInDegrees",
        ]
        p4 = get_keylist(gid, pkeys, "double")
        proj4["proj"] = "latlong"
        proj4["lon_wrap"] = round(sum(p4.values()) / 2.0)

        # use lon_wrap (center longitude) to make sure all values fall in same interval
        # if min_lon = -180: lon_wrap=0 (default)
        # if min_lon = 0: lon_wrap = 180 !!!
        # but you must also be able to handle e.g. [-20,270]

    elif gridtype == "rotated_ll":
        pkeys = [
            "angleOfRotationInDegrees",
            "latitudeOfSouthernPoleInDegrees",
            "longitudeOfSouthernPoleInDegrees",
        ]
        sp_lat = pkeys["latitudeOfSouthernPoleInDegrees"]
        sp_lon = pkeys["longitudeOfSouthernPoleInDegrees"]
        sp_angle = pkeys["angleOfRotationInDegrees"]
        if sp_angle != 0:
            # TODO: I don't know how to handle this. Need examples!
            logger.error("Rotated LatLon with SPangle not supported yet.")
        proj4["proj"] = "ob_tran"
        proj4["o_proj"] = "latlong"
        proj4["o_lat_p"] = -sp_lat
        proj4["o_lon_p"] = 0.0
        proj4["lon_0"] = sp_lon
    #  [rotated] mercator?
    else:
        return None
    return proj4


def get_gridinfo(gid):
    """Read all grid details and return all necessary data.

    Args:
        gid: GRIB handle

    Returns:
        a dictionary with the main grid characteristics
    """
    gridtype = get_keylist(gid, ["gridType"], "string")["gridType"]
    gkeys = [
        "Nx",
        "Ny",
        "uvRelativeToGrid",
        "iScansPositively",
        "jScansPositively",
        "latitudeOfFirstGridPointInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees",
        "DxInMetres",
        "DyInMetres",
        "iDirectionIncrementInDegrees",
        "jDirectionIncrementInDegrees",
    ]
    gg = get_keylist(gid, gkeys, "double")

    result = {
        "nlon": int(gg["Nx"]),
        "nlat": int(gg["Ny"]),
        "dx": gg["DxInMetres"],
        "dy": gg["DyInMetres"],
        "lon0": gg["longitudeOfFirstGridPointInDegrees"],
        "lat0": gg["latitudeOfFirstGridPointInDegrees"],
        "wrap_x": False,
    }
    if gridtype in ["regular_ll", "rotated_ll"]:
        result["dx"] = gg["iDirectionIncrementInDegrees"]
        result["dy"] = gg["jDirectionIncrementInDegrees"]
        logger.debug("%i vs %i", abs(360 - result["dx"] * result["nlon"]), result["dx"])
        if abs(360 - result["dx"] * result["nlon"]) < result["dx"]:
            result["wrap_x"] = True

    if not gg["iScansPositively"]:
        result["lon0"] = gg["longitudeOfLastGridPointInDegrees"]
    if not gg["jScansPositively"]:
        result["lat0"] = gg["latitudeOfLastGridPointInDegrees"]
    return result


def get_grid_limits(gid):
    """Find bounding box (lat/lon) of a grid."""
    lons, lats = get_grid_boundary(gid)

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


def get_keylist(gid, keylist, ktype="string"):
    """Get list of grib keys. Return "" or None for keys that are not found.

    Args:
        gid: GRIB handle
        keylist: list of key names
        ktype: required output type ("string", "double" or "long")

    Returns:
        A dictionary with all key values. Keys that are not present in the grib record
        return None.
    """
    if ktype == "string":
        func = eccodes.codes_get_string
        miss = ""
    elif ktype == "long":
        func = eccodes.codes_get_long
        miss = None
    elif ktype == "double":
        func = eccodes.codes_get_double
        miss = None

    ginfo = {}
    for kk in keylist:
        try:
            val = func(gid, kk)
        except eccodes.KeyValueNotFoundError:
            val = miss
        ginfo[kk] = val
    return ginfo


def param_match(gid, parameter_list):
    """Check whether a grib record is in the list of required parameters.

    TODO: can we re-organise the code to avoid getting the same key multiple times?
      But I suspect the impact is minimal

    Args:
        gid: grib handle
        parameter_list: list of parameter descriptors

    Returns:
        Parameter descriptor (from the list) that matches the current grib handle,
        or None.
    """
    for param in parameter_list:
        # param['grib_id'] is a dictionary of keys|values that describe the parameter
        # so you need to check all of these to know if the grib record matches
        plist = param["grib_id"]
        if isinstance(plist, dict):
            plist = [plist]

        for par in plist:
            keylist = list(par.keys())
            ginfo = get_keylist(gid, keylist, "string")
            # we make a deepcopy, because we may have to modify some key values...
            if match_keys(par, ginfo):
                pmatch = deepcopy(param)
                # make sure we have a single grib_id in the pmatch!
                pmatch["grib_id"] = deepcopy(par)
                pmatch["units"] = get_keylist(gid, ["parameterUnits"], "string")[
                    "parameterUnits"
                ]
                # make sure level information is explicitly passed!
                # e.g. for "2t" it may not be in the grib_id list.
                level_type = get_keylist(gid, ["typeOfLevel"], "string")["typeOfLevel"]

                pmatch["typeOfLevel"] = level_type
                pmatch["level"] = int(get_keylist(gid, ["level"], "long")["level"])
                if level_type == "isobaricInhPa":
                    pmatch["level_name"] = "p"
                elif level_type == "heightAboveGround":
                    pmatch["level_name"] = "z"
                elif level_type == "hybridLevel":
                    pmatch["level_name"] = "ml"
                elif level_type == "surface":
                    pmatch["level_name"] = None  #  'level' #  "sfc"
                else:
                    pmatch["level_name"] = None  # "level" #  "xx"

                return pmatch

    # no match was found
    return None


def points_restrict(gid, plist):
    """Restrict the station list to points inside the current domain.

    NOTE: * eccodes returns distances in kilometer
          * store the list for use in next run? run this function seperately?

    Args:
        gid: grib handle
        plist: list of stations

    Returns:
        reduced station list that contains only stations inside the domain
    """
    # 1. Get the bounding box lat/lon values
    #    that is a fast way to eliminate most outside points

    minlon, maxlon, minlat, maxlat = get_grid_limits(gid)
    gridinfo = get_gridinfo(gid)
    nlon = gridinfo["nlon"]
    nlat = gridinfo["nlat"]

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

    i, j = get_gridindex(lon, lat, gid)
    if gridinfo["wrap_x"]:
        # if x wraps around the globe, don't restrict at all
        # BUT: make sure to take this into account when interpolating!
        # NOTE: Amundsen-Scott SP base has j==0 !
        # With current code, i, j == 0 are OK, but == nlat|nlon-1 needs work
        p2 = p1[(j >= 0) & (j < nlat - 1)].copy()
    else:
        p2 = p1[(i >= 0) & (i < nlon - 1) & (j >= 0) & (j < nlat - 1)].copy()
    return p2


def get_grid_points(gid):
    """Get all lat/lon co-ordinates of the grid points.

    Args:
        gid: GRIB handle
    Returns:
        Numpy arrays with all lat/lon values.
    """
    gridinfo = get_gridinfo(gid)
    p4 = get_proj4(gid)
    proj = Proj(p4)
    nlon = int(gridinfo["nlon"])
    nlat = int(gridinfo["nlat"])
    dx = gridinfo["dx"]
    dy = gridinfo["dy"]

    x0, y0 = proj(gridinfo["lon0"], gridinfo["lat0"])

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


def get_grid_boundary(gid):
    """Get lat/lon co-ordinates of the grid boundary points.

    Args:
        gid: GRIB handle
    Returns:
        Numpy arrays with lat/lon values.
    """
    gridinfo = get_gridinfo(gid)
    p4 = get_proj4(gid)
    proj = Proj(p4)
    nlon = int(gridinfo["nlon"])
    nlat = int(gridinfo["nlat"])
    dx = gridinfo["dx"]
    dy = gridinfo["dy"]
    # get SW corner
    x0, y0 = proj(gridinfo["lon0"], gridinfo["lat0"])
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


def get_gridindex(lon, lat, gid):
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
    gridinfo = get_gridinfo(gid)
    p4 = get_proj4(gid)
    proj = Proj(p4)
    x0, y0 = proj(gridinfo["lon0"], gridinfo["lat0"])
    dx = gridinfo["dx"]
    dy = gridinfo["dy"]

    x, y = proj(np.array(lon), np.array(lat))
    i = (x - x0) / dx
    j = (y - y0) / dy
    return i, j


def train_weights(station_list, gid, lsm=False):
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
    i, j = get_gridindex(lon, lat, gid)

    gridinfo = get_gridinfo(gid)
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
    if gridinfo["wrap_x"]:
        logger.info("The domain wraps around the globe.")
        ic[ic == gridinfo["nlon"]] = 0
        ic[ic == -1] = gridinfo["nlon"]
        i1[i1 == gridinfo["nlon"]] = 0
        i0[i0 == -1] = gridinfo["nlon"]

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


def combine_fields(param, station_list):
    """Combine multiple decoded fields into one parameter.

    We need lat/lon and proj4 values when calculating wind speed & direction.
    Because we should correct for the local rotation of the axes.

    Args:
        param: full parameter descriptor
        station_list: a pandas table with at least lon & lat columns

    Returns:
        a new data matrix that combines the input fields according to parameter descriptor
    """
    nfields = len(param["data"])
    parname = param["harp_param"]
    if "level" in param:
        # NOTE: adding level to the name may sometimes lead to strange results
        #       e.g. S10m10, but it's just for debug messages, so we don't care.
        parname += str(param["level"])

    if any(dd is None for dd in param["data"]):
        # there are missing fields
        # don't make this a "warning", it would show too often...
        logger.info("COMBINE : missing data fields for parameter %s.", parname)
        return None

    logger.debug("COMBINING %i fields for %s", nfields, parname)

    npoints = len(param["data"][0])
    result = param_apply_function(param)


def parse_parameter_list(param_list, type="GRIB"):
    """Parse a (json) structure of required parameters.

    Handle single vs combined fields, model levels etc.

    Args:
        param_list: the list pf paramers (from json file)

    Returns:
        param_id_list: the complete list of fields that should be decoded
        param_combine: details for combined fields
    """
    param_sgl_list = []
    param_cmb_list = []
    if type == "GRIB":
        for param in param_list:
            if isinstance(param["grib_id"], dict):
                # a single field is decoded for this parameter
                if "level" in param["grib_id"] and isinstance(
                    param["grib_id"]["level"], list
                ):
                    # expand to multiple fields
                    for lev in param["grib_id"]["level"]:
                        pid = deepcopy(param)
                        pid["grib_id"]["level"] = str(lev)
                        param_sgl_list.append(pid)
                else:
                    pid = deepcopy(param)
                    # single field parameters are not cached for now
                    # so "data" entry is not needed
                    param_sgl_list.append(pid)

            else:
                # the parameter requires multiple fields
                # so we will have to cache the data and calculate later
                # for e.g. wind fields we must also cache grid and projection settings
                nfields = len(param["grib_id"])
                if "grib_id_common" not in param:
                    # the most simple combine case: no common keys
                    pid = deepcopy(param)
                    pid["data"] = [None] * nfields
                    pid["gridinfo"] = None
                    pid["proj4"] = None
                    param_cmb_list.append(pid)

                elif "level" in param["grib_id_common"] and isinstance(
                    param["grib_id_common"]["level"], list
                ):
                    # there is a common level key, so multiple entries
                    for lev in param["grib_id_common"]["level"]:
                        pid = deepcopy(param)
                        pid["grib_id_common"]["level"] = str(lev)
                        for f in range(len(pid["grib_id"])):
                            pid["grib_id"][f].update(pid["grib_id_common"])
                        pid.pop("grib_id_common")
                        pid["data"] = [None] * nfields
                        pid["gridinfo"] = None
                        pid["proj4"] = None
                        pid["level"] = lev
                        param_cmb_list.append(pid)

                else:
                    # there are common keys, but no level expansion
                    pid = deepcopy(param)
                    for f in range(len(pid["grib_id"])):
                        pid["grib_id"][f].update(pid["grib_id_common"])
                    pid.pop("grib_id_common")
                    pid["data"] = [None] * nfields
                    pid["gridinfo"] = None
                    pid["proj4"] = None
                    param_cmb_list.append(pid)

    elif type == "FA":
        for param in param_list:
            if isinstance(param["fa_id"], str):
                # a single field is decoded for this parameter
                # but possibly multiple levels
                if "fa_level" in param and isinstance(param["fa_level"], list):
                    # expand to multiple fields
                    for lev in param["fa_level"]:
                        pid = deepcopy(param)
                        pid["fa_id"] = fa_fix_level(param["fa_id"], lev)
                        param_sgl_list.append(pid)
                else:
                    pid = deepcopy(param)
                    # single field parameters are not cached for now
                    # so "data" entry is not needed
                    param_sgl_list.append(pid)
            else:
                # the parameter requires multiple fields
                # so we will have to cache the data and calculate later
                # for e.g. wind fields we must also cache grid and projection settings
                nfields = len(param["fa_id"])
                if "fa_level" not in param:
                    # the most simple combine case: no multiple levels
                    pid = deepcopy(param)
                    pid["data"] = [None] * nfields
                    pid["gridinfo"] = None
                    pid["proj4"] = None
                    param_cmb_list.append(pid)

                else:
                    # there is a common level key, so multiple entries
                    for lev in param["fa_level"]:
                        pid = deepcopy(param)
                        for f in range(len(pid["fa_id"])):
                            pid["fa_id"][f] = fa_fix_lev(pid["fa_id"][f], lev)
                        pid["data"] = [None] * nfields
                        pid["gridinfo"] = None
                        pid["proj4"] = None
                        pid["level"] = lev
                        param_cmb_list.append(pid)

    return param_sgl_list, param_cmb_list


def match_keys(p1, p2):
    """Match two sets of grib key values.

    Note that a key missing in p1 may be None in p2.

    Args:
      p1: a dictionary with GRIB2 key values
      p2: a dictionary with GRIB2 key values

    Returns:
      True if all (not None) values are equal. False otherwise.
    """
    for k in list(set(list(p1.keys()) + list(p2.keys()))):
        if k not in p1:
            if p2[k] is not None:
                return False
        elif k not in p2:
            if p1[k] is not None:
                return False
        elif p1[k] != p2[k]:
            return False
    return True


def cache_field(param, data, param_cmb_list, gid):
    """Check if a decoded field needs to be cached for later parameters.

    Args:
      param: parameter description of the current data
      data: decode (and interpolated) data
      param_cmb_list: a list (may be MODIFIED!)
      gid: GRIB handle (needed to cache the projection and grid details for every field)

    Returns:
      A count of the number of matching parameters.
      Usually 0 or 1, but wind components may be used for direction and speed,
      so may be matched twice.
    """
    count = 0
    gridinfo_list = ["uvRelativeToGrid"]
    for cmb in param_cmb_list:
        nfields = len(cmb["grib_id"])
        for ff in range(nfields):
            if match_keys(cmb["grib_id"][ff], param["grib_id"]):
                logger.debug("Caching output for %s", cmb["harp_param"])
                cmb["data"][ff] = np.array(data)
                # FIXME: we assume the unit is the same for all constituents and result
                #        this is NOT the case for e.g. wind direction
                #        probably should be fixed in the final combination function
                cmb["units"] = param["units"]
                cmb["level"] = param["level"]
                cmb["level_name"] = param["level_name"]
                if cmb["gridinfo"] is None:
                    cmb["gridinfo"] = get_keylist(gid, gridinfo_list, "long")
                    cmb["proj4"] = get_proj4(gid)
                count += 1
                continue
    logger.debug("Found %i matching combined parameters.", count)
    return count


def parse_grib_file(
    infile,
    param_list=None,
    station_list=None,
    sqlite_template=default_sqlite_template,
    weights=None,
    model_name="TEST",
):
    """Read a GRIB2 file and extract all required data points to SQLite.

    Args:
      infile: input grib2 file
      param_list: list of parameters or name of a json file
      station_list: pandas table with all stations or name of a csv file
      sqlite_template: template for sqlite output files
      weights: interpolation weights (if None, they are calculated)
      model_name: model name (string) used in the SQLite file name and data columns

    Returns:
        Total number of GRIB records and number of matching parameters found.
    """
    station_list = read_station_list(station_list)
    param_list = read_param_list(param_list)

    # split into "combined" and "direct" parameters
    param_sgl_list, param_cmb_list = parse_parameter_list(param_list)
    param_cmb_cache = [None] * len(param_cmb_list)
    if len(param_cmb_list) > 0:
        for pp in range(len(param_cmb_cache)):
            param_cmb_cache[pp] = {}

    logger.info(
        "SQLITE: expecting %i single and %i combined parameters.",
        len(param_sgl_list),
        len(param_cmb_list),
    )

    fcdate = None
    leadtime = None
    gt = 0
    gi = 0
    gd = 0
    gc = 0
    error_occured = False

    with open(infile, "rb") as gfile:
        while True:
            # loop over all grib records in the file
            gid = eccodes.codes_grib_new_from_file(gfile)
            if gid is None:
                # We've reached the last grib record in the file
                break
            gt += 1
            param = param_match(gid, param_sgl_list)
            if param is not None:
                direct = True
                gd += 1
                logger.debug("SQLITE: found parameter %s", param["harp_param"])
            else:
                # the record is not in our "direct" parameter list
                param = param_match(gid, param_cmb_list)
                if param is None:
                    eccodes.codes_release(gid)
                    continue
                # the record is in the "combined" list
                direct = False
                gc += 1
                logger.debug(
                    "SQLITE: found combinded parameter %s:%s",
                    param["harp_param"],
                    param["grib_id"],
                )

            # we have a matching parameter
            gi += 1
            # NOTE: actually, we only need to get fcdate & lead time once
            #       and we could even consider those to be known
            fcdate, leadtime = get_date_info(gid)
            logger.debug(
                "SQLITE: gt=%i gi=%i fcdate=%s, leadtime=%i, direct=%s",
                gt,
                gi,
                fcdate.isoformat(),
                leadtime,
                direct,
            )

            if weights is None:
                # We assume that station list and weights are the same for all GRIB fields
                # so we only "train" once
                # First reduce the station list to points inside the domain
                nstation_orig = station_list.shape[0]
                station_list = points_restrict(gid, station_list)
                if station_list.shape[0] == 0:
                    # In this case, we can not extract any points
                    logger.warning("SQLite: no stations inside model domain!")
                    error_occured = True
                    eccodes.codes_release(gid)
                    break

                logger.info(
                    "SQLITE: selected %i stations inside domain from %i.",
                    station_list.shape[0],
                    nstation_orig,
                )
                # create a list of interpolation weights
                logger.debug("SQLITE: training interpolation weights.")
                weights = train_weights(station_list, gid, lsm=False)

            # by default, we do bilinear interpolation
            method = param["method"] if "method" in param else "linear"
            # add columns to data table
            field_data = get_grid_values(gid)
            data_vector = interp_from_weights(field_data, weights, method)

            # cache this data vector if necessary
            # NOTE: a "direct" field may also be part of a combined field
            #       so we can not be sure without checking explicitly.
            # we may need to add some encoding information
            # like projection & uvRelativeToGrid for wind
            # so we pass gid along as well
            cache_field(param, data_vector, param_cmb_list, gid)

            # if this is a "direct" field, create a table and write to SQLite
            if direct:
                sqlite_file = sqlite_name(param, fcdate, model_name, sqlite_template)
                # NOTE: the column name for data gets an extra "_det"
                #       as required by HARP
                data = create_table(
                    data_vector,
                    station_list,
                    param,
                    fcdate,
                    leadtime,
                    model_name + "_det",
                )
                write_to_sqlite(data, sqlite_file, param, model_name + "_det")

            eccodes.codes_release(gid)

    # at this point, all grib records have been parsed (or an error occured)
    if error_occured:
        logger.error("An error occured. Exiting.")
        return gt, gi

    # OK, we have parsed the whole file and written all "direct" parameters
    # So now we still need to check all the "combined" ones
    # NOTE: this should only be run if we found some parameters in the first loop
    #       but checking whether the combined field is None is enough
    logger.debug("SQLITE: checking cached combined fields")
    for param in param_cmb_list:
        data_vector = combine_fields(param, station_list)
        # only write to SQLite if ALL components were found!
        if data_vector is not None:
            logger.debug("SQLITE: writing combined field")
            sqlite_file = sqlite_name(param, fcdate, model_name, sqlite_template)
            data = create_table(
                data_vector, station_list, param, fcdate, leadtime, model_name + "_det"
            )
            write_to_sqlite(data, sqlite_file, param, model_name + "_det")
    logger.info(
        "SQLITE: Total %i records. %i matching of which %i direct and %i combined.",
        gt,
        gi,
        gd,
        gc,
    )
    # Return: total count and # of matching param
    return gt, gi

def parse_fa_file(
    infile,
    param_list=default_parameter_list,
    station_list=default_station_list,
    sqlite_template=default_sqlite_template,
    model_name="TEST",
    weights = None
    ):

    """Read a FA file and extract all required data points to SQLite.

    Args:
      infile: input FAfile
      param_list: list of parameters or name of a json file
      station_list: pandas table with all stations or name of a csv file
      sqlite_template: template for sqlite output files
      model_name: model name (string) used in the SQLite file name and data columns
      weights: interpolation weights (NOT USED FOR FA DATA)

    Returns:
        Total number of FA records and number of matching parameters found.
    """
    if isinstance(station_list, str):
        station_list = read_station_list(station_list)
    if isinstance(param_list, str):
        param_list = read_param_list(param_list)

    # split into "combined" and "direct" parameters
    param_sgl_list, param_cmb_list = parse_parameter_list(param_list)
    param_cmb_cache = [None] * len(param_cmb_list)
    if len(param_cmb_list) > 0:
        for pp in range(len(param_cmb_cache)):
            param_cmb_cache[pp] = {}

    logger.info(
        "SQLITE: expecting %i single and %i combined parameters.",
        len(param_sgl_list),
        len(param_cmb_list),
    )

    gi = 0
    gd = 0
    gc = 0
    error_occured = False

    with epygram.open(infile, "r") as fafile:
        # WARNING: epygram returns field names without trailing blanks!
        field_list = fafile.listfields()
        gt = len(field_list)
        geo = fafile.geometry
        fcdate = fafile.validity[0]._basis # _basis _date_time 
        valdate = fafile.validity[0]._date_time
        leadtime = (valdate - fcdate).seconds

        logger.debug("fcdate = %s \n leadtime = %s", fcdate, leadtime)
        logger.debug("geo = %s", geo)

        nstation_orig = station_list.shape[0]
        station_list = points_restrict_fa(fafile, station_list)
        nstation_select = station_list.shape[0]
        if station_list.shape[0] == 0:
            # In this case, we can not extract any points
            logger.warning("SQLite: no stations inside model domain!")
            error_occured = True
            #fafile.close()
            return(gt,gi)

    # TODO: actually should loop over the union of single fields and combined
        param_sgl_names = set([ p['fa_id'] for p in param_sgl_list])
        param_cmb_names = set([ x for p in param_cmb_list for x in p['fa_id']])
        all_names = set.union(param_sgl_names, param_cmb_names)

        for pname in all_names:
            if pname in field_list:
                logger.debug("SQLITE: found parameter %s", pname)
            else:
                logger.debug("SQLITE: parameter %s not available", pname)
                continue
            gi += 1

            sgl_list = [ p for p in param_sgl_list if p['fa_id'] == pname ]
            cmb_list = [ p for p in param_cmb_list if pname in p['fa_id'] ]

            # you could in theory have multiple single vairables depending on 1 FA field
            # probably very rare...
            # NOTE: we will assume that for a given FA field, the interpolation method
            #       is always the same, so we only do the interpolation once.
            if len(sgl_list) > 0:
                method = sgl_list[0]['method']
            else:
                method = cmb_list[0]['method']

            field_data = fafile.readfield(pname)
            if field_data.spectral:
                field_data.sp2gp()

            # add columns to data table
            data_vector = field_data.getvalue_ll(
                lon=station_list["lon"],
                lat=station_list["lat"],
                interpolation=method)

            for param in sgl_list:
                gd += 1
                sqlite_file = sqlite_name(param, fcdate, model_name, sqlite_template)
                # NOTE: the column name for data gets an extra "_det"
                #       as required by HARP
                if 'function' in param:
                    dvect = call_function(param['function'], data_vector)
                else:
                    dvect = data_vector
                data = create_table(
                        dvect,
                        station_list,
                        param,
                        fcdate,
                        leadtime,
                        model_name + "_det",
                    )
                write_to_sqlite(data, sqlite_file, param, model_name + "_det")

            # if the field also occurs in combined fields (can be more than one!), put in cache
            for param in cmb_list:
                gc += 1
                cache_field_fa(param, data_vector, param_cmb_list, geo)

    for param in param_cmb_list:
        data_vector = combine_fields(param, station_list)
        # only write to SQLite if ALL components were found!
        if data_vector is not None:
            logger.debug("SQLITE: writing combined field")
            sqlite_file = sqlite_name(param, fcdate, model_name, sqlite_template)
            data = create_table(
                data_vector, station_list, param, fcdate, leadtime, model_name + "_det"
            )
            write_to_sqlite(data, sqlite_file, param, model_name + "_det")
    
    logger.info(
        "SQLITE: Total %i records. %i matching of which %i direct and %i combined.",
        gt,
        gi,
        gd,
        gc,
    )
    # Return: total count and # of matching param
    return gt, gi


