import epygram
import numpy as np
from .phys_functions import  combine_fields
from .sqlite_tools import *
#from epygram.geometries.VGeometry import hybridP2pressure, hybridP2altitude
from copy import deepcopy


from .logger import logger

# is this necessary? Doc is unclear...
epygram.init_env()

def get_proj4_fa(fafile):
    geoid = fafile.geometry.geoid # dict with a and b axes
    # for normal FA files:
    proj4 = {"R": 6371229}

    if fafile.geometry.name == "lambert":
        proj4["proj"] = "lcc"
        proj4["lon_0"] = fafile.geometry.projection['reference_lon'].get('degrees')
        proj4["lat_1"] = fafile.geometry.projection['reference_lat'].get('degrees')
        proj4["lat_2"] = proj4['lat_1']
    return proj4


def fa_expand_levels(fa_names, fa_levels):
    # both entries are expected to be lists or range
    lev_type = fa_names[0][0]
    lev_len =  fa_names.rfind("?") - fa_names.find("?") + 1
    if lev_type in ["P"]:
        fa_levels = [ (lev * 100) % 100000 for lev in fa_levels ]
    result = [ fanam.replace("?"*lev_len, f"{lev:0{lev_len}}")
            for lev in fa_levels
            for fanam in fa_names
            ]
    return(result)

def points_restrict_fa(fafile, plist):
    # Keep only stations inside the resource domain
    # FIXME: it appears that "CI" zone may be problematic
    #        some points are labelled "inside" but can not
    #        be interpolated to.
    is_in = fafile.geometry.point_is_inside_domain_ll(
            lon = plist["lon"],
            lat = plist["lat"],
            subzone = "C" # "CI" may give stations too close to border?
            )
    p1 = plist[ is_in ].copy().reset_index(drop=True)
    return(p1)

def get_geo_fa(fafile):
    """Read all grid details and return all necessary data.

    Args:
        fafile: FA filehandle

    Returns:
        a dictionary with the main domain characteristics
    """
#    geo_full = fafile.geometry
    if not fafile.geometry.rectangular_grid:
        logger.warning("Not a rectangular grid! Global data untested.")

#    gridtype = get_keylist(gid, ["gridType"], "string")["gridType"]
    # NOTE: we treat only the CI zone of the field
    result = {
        "nlon": int(fafile.geometry.dimensions["X_CIzone"]),
        "nlat": int(fafile.geometry.dimensions["Y_CIzone"]),
        "dx": fafile.geometry.grid['X_resolution'],
        "dy": fafile.geometry.grid['Y_resolution'],
        "lon0": fafile.geometry.gimme_corners_ll('CI')['ll'][0],
        "lat0": fafile.geometry.gimme_corners_ll('CI')['ll'][1],
        "minlat":fafile.geometry.minmax_ll('CI')['latmin'],
        "maxlat":fafile.geometry.minmax_ll('CI')['latmax'],
        "minlon":fafile.geometry.minmax_ll('CI')['lonmin'],
        "maxlon":fafile.geometry.minmax_ll('CI')['lonmax'],
        # FIXME
        "rotate_wind":True,
        "wrap_x": False,
        "grid_levels": fafile.geometry.vcoordinate.grid['gridlevels'],
    }
#    if gridtype in ["regular_ll", "rotated_ll"]:
#        result["dx"] = gg["iDirectionIncrementInDegrees"]
#        result["dy"] = gg["jDirectionIncrementInDegrees"]
#        logger.debug("%i vs %i", abs(360 - result["dx"] * result["nlon"]), result["dx"])
#        if abs(360 - result["dx"] * result["nlon"]) < result["dx"]:
#            result["wrap_x"] = True
    result['proj4'] = get_proj4_fa(fafile)

    return result

def parse_parameter_list_fa(param_list, nlev, nstat):
    """Parse a (json) structure of required parameters.

    Handle single vs combined fields, model levels etc.

    Args:
        param_list: the list pf paramers (from json file)
        nlev: the number of model levels (used in vertical interpolations)
        nstat: number of stations

    Returns:
        param_id_list: the complete list of fields that should be decoded
        param_combine: details for combined fields
    """
    # TODO: make it as type-agnostic as possible?
    #       netcdf...
    param_sgl_list = []
    param_cmb_list = []
    # TODO: restrict to allowed keys for FA: harp_param, method, fa_id, level, function, unit
    for param in param_list:
        if "hybrid_to_p" in param:
            # vertical interpolations, so we must get all hybrid levels
            # also, this data must be in np.array structures per field
            # NOTE: fa_id is always a list (because of SURFPRESSION)
            #       but fa_id_ml *can* be a single string, e.g. temperature
            param['fa_id_ml'] = param['fa_id'] # may be useful later...
            param['fa_id'] = fa_expand_levels(param['fa_id'], range(1, nlev + 1)) + ["SURFPRESSION"]

        if isinstance(param["fa_id"], str) or len(param["fa_id"] == 1):
            # a single field is decoded for this parameter
            # but possibly multiple levels
            if "level" in param and isinstance(param["level"], list):
                # FIXME: how should this be structured?
                #        SINGLE field at MULTIPLE LEVELS
                # expand to multiple fields
                # hybrid_to_p should NOT have a level key !!!
                if param["fa_id"][0] == "P":
                    param['level_name'] = "p"
                elif param["fa_id"][0] == "S":
                    #NOTE: "SURF" is OK, because then "level" should be undefined
                    param['level_name'] = "ml"
                elif param["fa_id"][0] == "H":
                    param['level_name'] = "z"
                else:
                    # NOTE: only p, z, ml are important
                    param['level_name'] = None

                pid = deepcopy(param)
                pid['data'] = np.full((1, nlev, nstat), np.nan)
                #for lev in param["level"]:
                #    pid = deepcopy(param)
                #    pid["fa_id"] = fa_fix_level(param["fa_id"], lev)
                #    param_sgl_list.append(pid)
            else:
                # SINGLE FIELD and SINGLE level
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
            if "hybrid_to_p" in param:
                # Special case: all the "levels" are to be combined and interpolated vertically
                # So we need to reserve enough cache space for all model levels in an np.array
                # We already expanded (to add surf pressure etc)
                #param['fa_id'] = fa_expand_3d_names( param['fa_id'], nlev)
                param["level_name"] = "p"
                pid = deepcopy(param)
                nfields = len(param['fa_id_ml'])
                # NOTE: we put all fields in a single 3d np.array, for faster interpolations
                #       the field order can be seen from first level.
                pid['surf_pressure'] = None
                pid['data'] = np.full((nfields, nlev, nstat), np.nan)
                pid["geo"] = None
                param_cmb_list.append(pid)

            elif "level" not in param:
                nfields = len(param["fa_id"])
                # the most simple combine case: no multiple levels
                param['level'] = None
                if "level_name" not in param:
                    param['level_name'] = None
                pid = deepcopy(param)
                # NOTE: we keep the "level" dimension, even if it is only 1
                #       for consistency through all possibilities
                pid["data"] = np.full((nfields, 1, nstat), np.nan)
                pid["geo"] = None
                param_cmb_list.append(pid)

            else:
                # There is no vertical interpolation
                # We expect to find the vertical fields directly.
                # there is a common level key, so multiple entries possible.
                # NOTE: to guess the level type (pressure, ...)
                #       we should look at the field names
                #       We look for the first field name with "?" in it
                # TODO: would an np.array be better than a list of vectors?
                #       advantage would be that it is mode similar to the vertical interpolation case
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

                pid = deepcopy(param)
                pid["data"] = np.full((nfields, len(param['level']), nstat), np.nan)
                pid["fa_id"] = fa_fix_level(pid["fa_id"][f], param['level'])


    return param_sgl_list, param_cmb_list

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
        if pname not in cmb['fa_id']:
            continue
        count += 1
        logger.debug("Caching output for %s: %s", cmb["harp_param"], pname)
        if "hybrid_to_p" in cmb and pname == "SURFPRESSION":
            cmb['surf_pressure'] = np.array(data)
        elif "level" in cmb or "hybrid_to_p" in cmb:
            # we need to find the "field" and "level" dimension of data
            i = cmb['fa_id'].index(pname)
            nf = cmb['data'].shape[0]
            nl = cmb['data'].shape[1] # = len(param['level'])
            i_l = i // nf
            i_f = i % nf
            logger.debug("Dimensions: i_f=%i, i_l=%i", i_f, i_l)
            cmb['data'][i_f, i_l, :] = np.array(data)
        else:
            # no level given, just a few fields that will be combined in a function
            # but the data structure still has 3 dimensions
            i_f = cmb['fa_id'].index(pname)
            cmb['data'][i_f, 0, :] = np.array(data)

        if cmb["geo"] is None:
            cmb['geo'] = geo
    logger.debug("Found %i matching combined parameters.", count)
    return count


def parse_fa_file(
    infile,
    param_list,
    station_list,
    sqlite_template,
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

    gm = 0
    gd = 0
    gc = 0
    error_occured = False

    with epygram.open(infile, "r") as fafile:
        # split into "combined" and "direct" parameters
        # FIXME: I need to open the FA file before parsing the parameters,
        #        because 3d interpolation needs knowledge of NLEV, NSTAT before building cache.
        nstation_orig = station_list.shape[0]
        logger.info("SQLITE: original number of stations in list: %i", nstation_orig)
        station_list = points_restrict_fa(fafile, station_list)
        nstation_select = station_list.shape[0]
        if station_list.shape[0] == 0:
            # In this case, we can not extract any points
            logger.warning("SQLite: no stations inside model domain!")
            error_occured = True
            #fafile.close()
            return(gd, gc)
        logger.info(
                    "SQLITE: selected %i stations inside domain from %i.",
                    station_list.shape[0],
                    nstation_orig,
                )

        nlev = len(fafile.geometry.vcoordinate.levels)
        param_sgl_list, param_cmb_list = parse_parameter_list_fa(param_list, nlev, nstation_select)
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

    # TODO: actually should loop over the union of single fields and combined
        param_sgl_names = set([ p['fa_id'] for p in param_sgl_list])
        param_cmb_names = set([ x for p in param_cmb_list for x in p['fa_id']])
        all_names = set.union(param_sgl_names, param_cmb_names)
        weights = None

        for pname in all_names:
            if pname not in field_list:
                logger.info("SQLITE: parameter %s not available", pname)
                continue

            logger.debug("SQLITE: found parameter %s", pname)
            gm += 1

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
            #if n_station_select is None:
            #    station_list = points_restrict_fa(fafile, station_list)
            #    nstation_select = station_list.shape[0]
            #    if station_list.shape[0] == 0:
            #        # In this case, we can not extract any points
            #        logger.warning("SQLite: no stations inside model domain!")
            #        error_occured = True
            #        fafile.close()
            #        return(gd, gc)
            #    logger.info(
            #        "SQLITE: selected %i stations inside domain from %i.",
            #        station_list.shape[0],
            #        nstation_orig,
            #    )

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
                cache_field_fa(pname, data_vector, param_cmb_list, geo)

    for param in param_cmb_list:
        if "hybrid_to_p" in param:
            # TODO: check that there is no missing data?
            #       Or just trust that numpy will take care of it.
            grid_levels = param["geo"]["grid_levels"]
            A = [level[1]['Ai'] for level in grid_levels][1:]
            B = [level[1]['Bi'] for level in grid_levels][1:]
            target_p = param["hybrid_to_p"]

            surfP = param["surf_pressure"]
            if (surfP[0] < 50) surfP = np.exp(surfP)
            vertP = hybridP2masspressure(A, B, surfP, 'arithmetic') / 100
            # an array with the pressures for all ML's at every point

            # dimensions are [ field, level, location ]
            ML_data = param["data"][:,:,:]
            PL_data = log_interpolation(target_p, vertP, ML_data)

            param["level"] = target_p
            param["level_name"] = "p"
            param["data"] = PL_data

        data_vector = combine_fields(param, station_list)
        # only write to SQLite if ALL components were found!
        if data_vector is None:
            logger.info("Not able to build parameter %s .", param['harp_param'])
        else:
            logger.debug("SQLITE: writing combined field")
            sqlite_file = sqlite_name(param, fcdate, model_name, sqlite_template)
            data = create_table(
                data_vector, station_list, param, fcdate, leadtime, model_name
            )
            write_to_sqlite(data, sqlite_file, param, model_name)
            gc += 1

    logger.info(
        "SQLITE: Total %i records. %i matching of which %i direct and %i combined.",
        gt,
        gm,
        gd,
        gc,
    )
    # Return: total count and # of matching param
    return gd, gc


