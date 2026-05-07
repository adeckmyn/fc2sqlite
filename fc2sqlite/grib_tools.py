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

def get_geo_grib(gid):
    """Read all grid details and return all necessary data.

    Args:
        gid: GRIB handle

    Returns:
        a dictionary with the main domain characteristics
    """

    gridtype = get_keylist(gid, ["gridType"], "string")["gridType"]
    gkeys = [
        "Nx",
        "Ny",
        "latitudeOfFirstGridPointInDegrees",
        "longitudeOfFirstGridPointInDegrees",
        "latitudeOfLastGridPointInDegrees",
        "longitudeOfLastGridPointInDegrees",
        "iScansPositively",
        "jScansPositively",
        "DxInMetres",
        "DyInMetres",
        "iDirectionIncrementInDegrees",
        "jDirectionIncrementInDegrees",
        "uvRelativeToGrid",
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
        "rotate_wind":gg["uvRelativeToGrid"] == 1,
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

    result['proj4'] = get_proj4(gid)

    return result


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


