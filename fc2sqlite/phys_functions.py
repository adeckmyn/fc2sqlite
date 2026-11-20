import numpy as np
from epygram.geometries.VGeometry import hybridP2masspressure

from .logger import logger

#####################################
# UNITS AND CONSTANTS
#####################################

R_w = 461.52
R_d = 287
h_i = 2500000
g = 9.81
eps = R_d / R_w

#####################################
# PHYSICS
#####################################

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

def param_apply_function(param, station_list):
    nfields = len(param["data"])
    parname = param["harp_param"]

    if param['function'] == 'vector_angle':
        param["units"] = "deg"
        if nfields != 2:
            logger.error("ERROR: angle always needs exactly two components.")
        u = param["data"][0]
        v = param["data"][1]
        direction_to = np.degrees(np.arctan2(u, v))
        direction_from = (direction_to + 180) % 360
        if param["geo"]["rotate_wind"]:
        # FIXME: wind may need to be rotated first !!!
        #        also FA syntax!
            lat = np.array(station_list["lat"].tolist())
            lon = np.array(station_list["lon"].tolist())
            logger.debug("Correcting wind angle.")
            angle, mapfactor = rotate_wind(lon, lat, param["geo"]["proj4"])
            direction_from = direction_from + angle

        return direction_from

    elif param['function'] == 'vector_norm':
        npoints = len(param["data"][0])
        result = np.zeros(npoints)
        for ff in range(nfields):
            result += param["data"][ff] * param["data"][ff]
        return np.sqrt(result)

    elif param['function'] == 'sum':
        npoints = len(param["data"][0])
        result = np.zeros(npoints)
        for ff in range(nfields):
            result += param["data"][ff]
        return result

    elif param['function'] == "PQT_to_RH":
        if nfields != 3:
            logger.error("ERROR: RH from Q needs exactly 3 components.")
        P = param["data"][0]
        Q = param["data"][1]
        T = param["data"][2]
        return PQT_to_RH(P, Q, T)

    elif param['function'] == "PQT_to_Td":
        if nfields != 3:
            logger.error("ERROR: Td from Q,T needs exactly 3 components.")
        P = param["data"][0]
        Q = param["data"][1]
        T = param["data"][2]
        return PQT_to_Td(P, Q, T)

    elif param['function'] == "hybrid_to_p":
        # TODO
        logger.warning("3D interpolation is not yet implemented")
        return None

    else:
        logger.error("Unknown function %s.", param['function'])
        return None

def PQT_to_RH(P, Q, T):
    # Returns relative humidity (fraction, 0–1).
    # Inputs: pressure P (hPa), specific humidity Q (kg/kg), temperature T (K).
    # Uses Tetens formula for saturation vapor pressure (over water/ice).
    e = (Q * P) / (eps + (1.0 - eps) * Q)
    es = np.where(T > 0, 
            6.1078 * np.exp((17.27 * T) / (T + 237.3)),
            6.1078 * np.exp((21.875 * T) / (T + 265.5)))
    RH = e / es
    RH = np.clip(RH, 0.0, 1.0)
    return RH

def PQT_to_Td(P, Q, T):
    # Returns dew point temperature Td (K).
    # Inputs: pressure p (hPa), specific humidity Q (kg/kg), temperature T (K).
    # Computes relative humidity first, then applies Clausius–Clapeyron approximation.
    RH = PQT_to_RH(P, Q, T)
    return TRH_to_Td(T, RH)

def TTd_to_RH(T, Td):
    # Returns relative humidity (fraction, 0–1).
    # Inputs: temperature T (K), dew point temperature Td (K).
    # Computed using Clausius–Clapeyron relation assuming constant latent heat.
    RH = np.exp((h_i/R_w) * (1/T - 1/Td))
    RH = np.clip(RH, 0.0, 1.0)
    return RH

def PTTd_to_Q(P, T, Td):
    # Returns specific humidity q (kg/kg).
    # Inputs: pressure p (hPa), temperature T (K), dew point temperature Td (K).
    # Relative humidity is computed using Clausius–Clapeyron approximation,
    # then vapor pressure is obtained via Tetens formula for saturation vapor pressure.

    RH = Td_to_RH(T, Td)

    T_C = T - 273.15

    es = np.where(T_C > 0, 
            6.1078 * np.exp((17.27 * T_C) / (T_C + 237.3)),
            6.1078 * np.exp((21.875 * T_C) / (T_C + 265.5)))

    e = RH * es

    q = (eps * e) / (P - (1 - eps) * e)

    return q

def TRH_to_Td(T, RH):
    # Returns dew point temperature Td (K).
    # Inputs: temperature T (K), relative humidity RH (fraction, 0–1).
    # Uses Clausius–Clapeyron approximation (constant latent heat).
    RH = np.clip(RH, a_min=0.01, a_max=1.0)

    Td = 1 / (1/T - (R_w/h_i) * np.log(RH))

    return Td

def msl_reduction(p,z,T):
    # Returns mean sea level pressure (same units as p).
    # Inputs: surface pressure p, height z (m), temperature T (K).
    # Uses barometric formula assuming isothermal atmosphere.

    if np.all(T == 0):
        return p

    mslp = p * np.exp((g * z) / (R_d * T))

    return mslp

def circular_interpolation(target_pressure, pressure, angles):

    # Returns interpolated direction (degrees, 0–360).
    # Inputs: target pressure, pressure levels, angles (degrees).
    # Performs circular interpolation by unwrapping angles, interpolating in log-pressure space, then rewrapping.

    angles_rad = np.radians(angles)
    angles_unwrapped = np.unwrap(angles_rad)

    angles_interpolated = log_interpolation(target_pressure, pressure, angles_unwrapped)

    return np.degrees(angles_interpolated) % 360

def log_interpolation(x_target, x, y): 
    # TODO: let y  have multiple dimensions (field, level, station)
    #       x has pressures at all model levels for every station (level, station)
    #       while x_target has pressure levels as a single vector
    # Returns interpolated values y at x_target.
    # Performs linear interpolation in log(x) (i.e., y is linear in log(x)).
    # Requires x > 0 and x_target > 0.

    x_target = np.asarray(x_target)
    x = np.asarray(x)
    y = np.asarray(y)

    if np.any(x <= 0) or np.any(x_target <= 0):
        raise ValueError("log_interpolation: x and x_target must be > 0")

    idx = np.argsort(x)
    x = x[idx]
    y = y[idx]

    y_interp = np.interp(np.log(x_target), np.log(x), y)

    return y_interp


def rotate_wind(lon, lat, p4):
    """Rotate wind from projection grid to geographical axes for correct wind direction.

    Args:
        lon: longitude (numpy vectors or single value)
        lat: latitude (numpy vectors or single value)
        p4: proj4 definition as a list (not single string)

    Returns:
        angle: correction angle in deg (vector) corresponding to every lat/lon location.
    """
    if p4["proj"] == "lcc":
        rad = np.pi / 180.0
        refcos = np.cos(p4["lat_1"] * rad)
        refsin = np.sin(p4["lat_1"] * rad)
        angle = refsin * (lon - p4["lon_0"])
        mapfactor = np.power(refcos / np.cos(lat * rad), 1 - refcos) * np.power(
            (1 + refsin) / (1 + np.sin(lat * rad)), refsin
        )
    elif p4["proj"] == "latlong":
        angle = np.zeros(len(lat))
        mapfactor = np.ones(len(lat))
    else:
        logger.error("Unimplemented wind rotation for projection %s.", p4["proj"])
        angle = np.zeros(len(lat))
        mapfactor = np.ones(len(lat))

    return angle, mapfactor

def hybrid_to_p(param):
    grid_levels = param["geo"]["grid_levels"]
    target_p = param["hybrid_to_p"]

    A = [level[1]['Ai'] for level in grid_levels][1:]
    B = [level[1]['Bi'] for level in grid_levels][1:]

    surfP = param["data"][0]
    # is it log-pressure?
    if np.max(surfP) < 100 surfP = np.exp(surfP)
    # interpolate all separate fields
    # use this for all following variables!
    vertP = hybridP2masspressure(A, B, surfP, 'arithmetic') / 100
    # surfP is an array with local pressure at all ml's
    # first co-ordinate is ml, second is station
    
    for ff in param["data"][1:]:
        #paramdata = param["data"][:,1,:]

    result = log_interpolation(target_p, vertP, paramdata)

    param["level"] = target_p

    return result


