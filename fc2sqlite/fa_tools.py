import epygram
import numpy as np
#from epygram.geometries.VGeometry import hybridP2pressure, hybridP2altitude

from .logger import logger

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


def fa_fix_level(fa_name, fa_level):
    # expected fa_name: S???SOMETHING, P?????SOMETHING, H?????SOMETHING
    lev_type = fa_name[0] # S, P, H ...
    lev_len =  fa_name.rfind("?") - fa_name.find("?") + 1
    if lev_type in ["P"]:
        fa_level = (fa_level * 100) % 100000
    result = fa_name.replace("?"*lev_len, f"{fa_level:0{lev_len}}")
    return(result)


def points_restrict_fa(fafile, plist):
    # Keep only stations inside the resource domain
    is_in = fafile.geometry.point_is_inside_domain_ll(
            lon = plist["lon"],
            lat = plist["lat"],
            subzone = "CI"
            )
    p1 = plist[ is_in ].copy()
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
        "wrap_x": False,
    }
#    if gridtype in ["regular_ll", "rotated_ll"]:
#        result["dx"] = gg["iDirectionIncrementInDegrees"]
#        result["dy"] = gg["jDirectionIncrementInDegrees"]
#        logger.debug("%i vs %i", abs(360 - result["dx"] * result["nlon"]), result["dx"])
#        if abs(360 - result["dx"] * result["nlon"]) < result["dx"]:
#            result["wrap_x"] = True
    result['proj4'] = get_proj4_fa(fafile)

    return result

