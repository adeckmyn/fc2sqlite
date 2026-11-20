#! /usr/bin/env python3
"""fc2sqlite: toolbox for extracting point data from GRIB fields."""
from .fa_tools import parse_fa_file
from .grib_tools import parse_grib_file
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
defaults = {
        'parameter_list':"param_list_default.json",
        'station_list':"station_list_default.csv",
        'sqlite_template':"{MODEL}/{YYYY}/{MM}/FCTABLE_{PP}_{YYYY}{MM}.sqlite",
        'model_name':"TEST_det",
        }


# use epygram.formats.guess(filename) ?
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
        elif header[0:6] == b'netcdf':
            return("netCDF")
        else:
            faheader = list(struct.unpack(">5Q", header))
            if faheader[1] == 16 and faheader[3] == 22:
                return("FA")
    return(None)


def read_param_list(param_file = None, macros = {}):
    """Read a parameter file (json).

    Args:
        param_file: name of a json file

    Returns:
        A dict containing the parsed parameter list
    """
    if param_file is None:
        param_file = defaults['parameter_list']

    if not os.path.isfile(param_file) and os.path.dirname(param_file) == "":
        param_file = os.path.join(basedir, param_file)

    try:
        with open(param_file) as pf:
            parameter_list = json.load(pf)
            pf.close()
    except OSError:
        logger.error("Can not read parameter file %s", param_file)
        raise

    # MACRO substitutions

    if "macros" in parameter_list[0]:
        logger.info("Resolving macros in parameter list.")
        macros_conf = parameter_list.pop(0)['macros']
    else:
        macros_conf = {}
    # macros defined via command line get preference
    macros = { **macros_conf, **macros)

    if macros:
        parameter_list = substitute_macros(parameter_list, macros)

    return parameter_list


def substitute_macros(indict, macros)
    for key in indict.keys():
        if isinstance(indict[key], "dict"):
            indict[key] = substitute_macros(indict[key], macros)
        else:
            if indict[key] in macros:
                indict[key] = macros[indict[key]]
    return indict

def read_station_list(station_file = None):
    """Read a station file (csv).

    Args:
        station_file: name of a csv file

    Returns:
        A pandas table containing the parsed station list
    """
    if station_file is None:
        station_file = defaults['station_list']

    if not os.path.isfile(station_file) and os.path.dirname(station_file) == "":
        station_file = os.path.join(basedir, station_file)

    try:
        station_list = pandas.read_csv(station_file, skipinitialspace=True)
    except OSError:
        logger.error("Can not read station file %s", station_file)
        raise
    return station_list


def match_param(p1, p2):
    if "fa_id" in p1:
        return p1['fa_id'] == p2['fa_id']
    elif "grib_id" in p1:
        return match_keys(p1, p2)

def cache_field(param, data, param_cmb_list, geo):
    """Check if a decoded field needs to be cached for later parameters.

    Args:
      param: parameter description of the current data
      data: np array with interpolated data
      param_cmb_list: a list (may be MODIFIED!)
      geo: a dict with main geometry of the grid

    Returns:
      A count of the number of matching parameters.
      Usually 0 or 1, but wind components may be used for direction and speed,
      so may be matched twice.
    """
    count = 0
    fid = 'grib_id'
    #if isinstance(param, str):
    #    # if the parameter is just a string, it's an FA field name
    #    fid = "fa_id"
    #else:
    #    # for GRIB, the parameter is always a dict
    #    fid = "grib_id"

    #if param['type'] == 'GRIB':
    #    fid = 'grib_id'
    #elif param['type'] == 'FA':
    #    fid = 'fa_id'

    for cmb in param_cmb_list:
        nfields = len(cmb[fid])
        for ff in range(nfields):
            if match_keys(cmb[fid][ff], param[fid]):
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


def parse_fc_file(infile,
    param_list,
    station_list,
    sqlite_template=defaults['sqlite_template'],
    model_name=defaults['model_name'],
    weights=None):

    file_type = get_file_type(infile)

    if isinstance(station_list, str):
        station_list = read_station_list(station_list)
    if isinstance(param_list, str):
        param_list = read_param_list(param_list) 


    if file_type == "FA":
        gd, gc = parse_fa_file(infile, param_list, station_list, sqlite_template, model_name, weights)
    elif file_type == "GRIB":
        gd, gc = parse_grib_file(infile, param_list, station_list, sqlite_template, model_name, weights)
    elif file_type == "netCDF":
        gd, gc = parse_netcdf_file(infile, param_list, station_list, sqlite_template, model_name, weights)
    else:
        logger.error("Unknown file type %s", file_type)
        gd = gc = 0
    return gd, gc



