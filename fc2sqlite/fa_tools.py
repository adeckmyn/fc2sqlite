import epygram
import numpy as np
#from epygram.geometries.VGeometry import hybridP2pressure, hybridP2altitude

from .logger import logger

def fa_fix_level(fa_name, fa_level):
    # expected fa_name: S???SOMETHING, P?????SOMETHING, H?????SOMETHING
    lev_type = fa_name[0] # S, P, H ...
    lev_len =  fa_name.rfind("?") - fa_name.find("?") + 1
    if lev_type in ["P"]:
        fa_level = (fa_level * 100) % 100000
    fa_name[1:lev_len+1] = f"{fa_level:0{lev_len}}"
    return(fa_name)


def points_restrict_fa(fafile, plist):
    # Keep only stations inside the resource domain
    is_in = fafile.geometry.point_is_inside_domain_ll(
            lon = plist["lon"],
            lat = plist["lat"]
            )
    p1 = plist[ is_in ].copy()
    return(p1)


