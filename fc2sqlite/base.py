#! /usr/bin/env python3
"""fc2sqlite: toolbox for extracting point data from GRIB fields."""
from .fa_tools import *
from .grib_tools import *
from .sqlite_tools import sqlite_name, create_table, write_to_sqlite
from .phys_functions import param_apply_function
from .interpolate import *

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


def proj4_to_string(proj4):
    """Transfrom a proj4 from dictionary to string.

    Args:
        proj4: a dictionary

    Returns:
        a single string
    """
    result = " ".join([f"+{p}={proj4[p]}" for p in proj4])
    return result


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
    result = param_apply_function(param, station_list)
    return result


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

def parse_parameter_list_fa(param_list, nlev):
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
        if isinstance(param["fa_id"], str):
            # a single field is decoded for this parameter
            # but possibly multiple levels
            if "level" in param and isinstance(param["level"], list):
                # expand to multiple fields
                if param["fa_id"][0] == "P":
                    param['level_name'] = "p"
                elif param["fa_id"][0] == "S":
                    param['level_name'] = "ml"
                elif param["fa_id"][0] == "H":
                    param['level_name'] = "z"
                else:
                    # NOTE: only p, z, ml are important
                    param['level_name'] = None

                for lev in param["level"]:
                    pid = deepcopy(param)
                    pid["fa_id"] = fa_fix_level(param["fa_id"], lev)
                    param_sgl_list.append(pid)
            else:
                if "level" not in param:
                    param['level'] = None
                if "level_name" not in param:
                    param['level_name'] = None
                pid = deepcopy(param)
                # single field parameters are not cached for now
                # so "data" entry is not needed
                param_sgl_list.append(pid)
        else:
            # the parameter requires multiple fields
            # so we will have to cache the data and calculate later
            # for e.g. wind fields we must also cache grid and projection settings
            nfields = len(param["fa_id"])
            if "level" not in param:
                # the most simple combine case: no multiple levels
                param['level'] = None
                if "level_name" not in param:
                    param['level_name'] = None
                pid = deepcopy(param)
                pid["data"] = [None] * nfields
                pid["geo"] = None
                param_cmb_list.append(pid)

            elif param['function'] == "hybrid_to_p":
                # Special case: all the "levels" are to be combined and interpolated vertically
                # TODO: something better than caching all hybrid levels for every pressure...
                # FIXME: we need to know the number of model levels!
                param['fa_id'] = fa_expand_3d_names( param['fa_id'], nlev)
                nfields = len(param['fa_id'])
                pid = deepcopy(param)
                pid["data"] = [None] * nfields
                pid["geo"] = None
                param_cmb_list.append(pid)


            else:
                # there is a common level key, so multiple entries
                # NOTE: to guess the level type (pressure, ...)
                #       we should look at the field names
                #       We look for the first field name with "?" in it
                lev_field = [ x for x  in param['fa_id'] if "?" in x ][0]
                if lev_field[0] == "P":
                    param['level_name'] = "p"
                elif lev_field[0] == "S":
                    param['level_name'] = "ml"
                elif lev_field[0] == "H":
                    param['level_name'] = "z"
                else:
                    # NOTE: only p, z, ml are important
                    param['level_name'] = None

                if not isinstance(param["level"], list):
                    param["level"] = [ param["level"] ]

                for lev in param["level"]:
                    # FIXME: this fails if the field list also includes non-level fields like pressure
                    # basically, this is not a good approach for 3D fields (vertical interpolation)
                    pid = deepcopy(param)
                    for f in range(nfields):
                        pid["fa_id"][f] = fa_fix_level(pid["fa_id"][f], lev)
                    pid["data"] = [None] * nfields
                    pid["geo"] = None
                    pid["level"] = lev
                    logger.debug("COMBI LEVEL")
                    logger.debug(pid)
                    param_cmb_list.append(pid)

    return param_sgl_list, param_cmb_list


def cache_field(param, data, param_cmb_list, geo):
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
                if cmb["geo"] is None:
                    cmb["geo"] = geo
                count += 1
                continue
    logger.debug("Found %i matching combined parameters.", count)
    return count

def cache_field_fa(pname, data, param_cmb_list, geo):
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
    for cmb in param_cmb_list:
        nfields = len(cmb["fa_id"])
        for ff in range(nfields):
            if cmb["fa_id"][ff] == pname:
                logger.debug("Caching output for %s", cmb["harp_param"])
                cmb["data"][ff] = np.array(data)
                if cmb["geo"] is None:
                    cmb['geo'] = geo
                # FIXME: we assume the unit is the same for all constituents and result
                #        this is NOT the case for e.g. wind direction
                #        probably should be fixed in the final combination function
                # FIXME: these have to be done when creating the cache!
                #cmb["units"] = param["units"]
                #cmb["level"] = param["level"]
                #cmb["level_name"] = param["level_name"]
                #if cmb["geo"] is None:
                #    cmb["geo"] = geo
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
    param_sgl_list, param_cmb_list = parse_parameter_list_grib(param_list)
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
                    "SQLITE: found combined parameter %s:%s",
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


    gi = 0
    gd = 0
    gc = 0
    error_occured = False

    with epygram.open(infile, "r") as fafile:
        # split into "combined" and "direct" parameters
        # FIXME: I need to open the FA file before parsing the parameters,
        #        because 3d interpolation needs knowledge of NLEV before building cache.
        nlev = len(fafile.geometry.vcoordinate.levels)
        param_sgl_list, param_cmb_list = parse_parameter_list_fa(param_list, nlev)
        param_cmb_cache = [None] * len(param_cmb_list)
        if len(param_cmb_list) > 0:
            for pp in range(len(param_cmb_cache)):
                param_cmb_cache[pp] = {}

        logger.info(
            "SQLITE: expecting %i single and %i combined parameters.",
            len(param_sgl_list),
            len(param_cmb_list),
        )
        # WARNING: epygram returns field names without trailing blanks!
        field_list = fafile.listfields()
        gt = len(field_list)
        geo = get_geo_fa(fafile)
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
        logger.info(
                    "SQLITE: selected %i stations inside domain from %i.",
                    station_list.shape[0],
                    nstation_orig,
                )

    # TODO: actually should loop over the union of single fields and combined
        param_sgl_names = set([ p['fa_id'] for p in param_sgl_list])
        param_cmb_names = set([ x for p in param_cmb_list for x in p['fa_id']])
        all_names = set.union(param_sgl_names, param_cmb_names)
        weights = None

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
            # NOTE: interpolation via both methods is comparable in speed

            data_vector = field_data.getvalue_ll(
                    lon=station_list["lon"],
                    lat=station_list["lat"],
                    interpolation=method)
            #if weights is None:
            #        geo = get_geo_fa(fafile)
            #        #station_list = points_restrict(geo, station_list)
            #        weights = train_weights(station_list, geo, lsm=False)
            #data_vector = interp_from_weights(field_data.data, weights, method)

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
                # BUG: param has the list fa_id names!
                #      we should only send the current name
                #      but ideally also the other parameter settings?
                gc += 1
                cache_field_fa(pname, data_vector, param_cmb_list, geo)

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


