import os
import json
import struct
from copy import deepcopy
from datetime import datetime

import pandas
import eccodes
import numpy as np
from pyproj import Proj
from .phys_functions import combine_fields
from .sqlite_tools import *
from .interpolate import *
from copy import deepcopy


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

def parse_parameter_list_grib(param_list):
    """Parse a (json) structure of required parameters.

    Handle single vs combined fields, model levels etc.

    Args:
        param_list: the list pf paramers (from json file)

    Returns:
        param_id_list: the complete list of fields that should be decoded
        param_combine: details for combined fields
    """
    # TODO: make it as type-agnostic as possible?
    #       netcdf...
    param_sgl_list = []
    param_cmb_list = []
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
                pid["geo"] = None
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
                    pid["geo"] = None
                    pid["level"] = lev
                    param_cmb_list.append(pid)

            else:
                # there are common keys, but no level expansion
                pid = deepcopy(param)
                for f in range(len(pid["grib_id"])):
                    pid["grib_id"][f].update(pid["grib_id_common"])
                pid.pop("grib_id_common")
                pid["data"] = [None] * nfields
                pid["geo"] = None
                param_cmb_list.append(pid)
    return param_sgl_list, param_cmb_list

def parse_grib_file(
    infile,
    param_list,
    station_list,
    sqlite_template,
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

    # split into "combined" and "direct" parameters
    param_sgl_list, param_cmb_list = parse_parameter_list_grib(param_list)
    logger.info(
        "SQLITE: expecting %i single and %i combined parameters.",
        len(param_sgl_list),
        len(param_cmb_list),
    )

    fcdate = None
    leadtime = None
    gt = 0
    gm = 0
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
                logger.debug(
                    "SQLITE: found combined parameter %s:%s",
                    param["harp_param"],
                    param["grib_id"],
                )

            # we have a matching parameter
            gm += 1
            # NOTE: actually, we only need to get fcdate & lead time once
            #       and we could even consider those to be known
            fcdate, leadtime = get_date_info(gid)
            logger.debug(
                "SQLITE: gt=%i gm=%i fcdate=%s, leadtime=%i, direct=%s",
                gt,
                gm,
                fcdate.isoformat(),
                leadtime,
                direct,
            )

            if weights is None:
                # We assume that station list and weights are the same for all GRIB fields
                # so we only "train" once
                # First reduce the station list to points inside the domain
                nstation_orig = station_list.shape[0]
                geo = get_geo_grib(gid)
                station_list = points_restrict(geo, station_list)
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
                weights = train_weights(station_list, geo, lsm=False)

            # by default, we do bilinear interpolation
            method = "linear" if "method" not in param or param['method']=="bilin" else param["method"]

            # add columns to data table
            field_data = get_grid_values(gid)
            data_vector = interp_from_weights(field_data, weights, method)
            #The following gives a seg fault further on:
            #data_vector = epygram.formats.GRIB.GRIBmessage(source=('gribid', gid)).as_field().getvalue_ll(
            #        lon=station_list["lon"],
            #        lat=station_list["lat"],
            #        interpolation=method)

            # cache this data vector if necessary
            # NOTE: a "direct" field may also be part of a combined field
            #       so we can not be sure without checking explicitly.
            # we may need to add some encoding information
            # like projection & uvRelativeToGrid for wind
            # so we pass gid along as well
            cache_field(param, data_vector, param_cmb_list, geo)

            # if this is a "direct" field, create a table and write to SQLite
            if direct:
                sqlite_file = sqlite_name(param, fcdate, model_name, sqlite_template)
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
        return gd, gc

    # OK, we have parsed the whole file and written all "direct" parameters
    # So now we still need to check all the "combined" ones
    # NOTE: this should only be run if we found some parameters in the first loop
    #       but checking whether the combined field is None is enough
    logger.debug("SQLITE: checking cached combined fields")
    for param in param_cmb_list:
        data_vector = combine_fields(param, station_list)
        # only write to SQLite if ALL components were found!
        if data_vector is not None:
            gc += 1
            logger.debug("SQLITE: writing combined field")
            sqlite_file = sqlite_name(param, fcdate, model_name, sqlite_template)
            data = create_table(
                data_vector, station_list, param, fcdate, leadtime, model_name
            )
            write_to_sqlite(data, sqlite_file, param, model_name)
    logger.info(
        "SQLITE: Total %i records. %i matching of which %i direct and %i combined.",
        gt,
        gm,
        gd,
        gc,
    )
    # Return: number of "direct" and "combined" parameters found
    return gd, gc


