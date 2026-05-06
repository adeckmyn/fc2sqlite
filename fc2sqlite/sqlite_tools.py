import sqlite3
import os
from contextlib import closing
from .logger import logger
#import logging
#logger = logging.getLogger(__name__)


def sqlite_name(param, fcdate, model_name, sqlite_template):
    """Create the full name of the SQLite file from template and date.

    Args:
        param: parameter descriptor
        fcdate: forecast date
        model_name: model name (string) used in the SQLite file name and data columns
        sqlite_template: template for SQLite file name

    Returns:
        full name of SQLite file
    """
    result = sqlite_template
    result = result.replace("{PP}", param["harp_param"])
    result = result.replace("{YYYY}", fcdate.strftime("%Y"))
    result = result.replace("{MM}", fcdate.strftime("%m"))
    result = result.replace("{DD}", fcdate.strftime("%d"))
    result = result.replace("{HH}", fcdate.strftime("%H"))
    result = result.replace("{MODEL}", model_name)
    return result


def create_table(data_vector, station_list, param, fcdate, leadtime, model_name):
    """Put a data vector (interpolated field) to pandas table.

    Args:
      data_vector: numpy vector with interpolated values
      station_list: pandas table with station list
      param: parameter descriptor
      fcdate: forecast date (datetime object)
      leadtime: lead time (int)
      model_name: model name (string)

    Returns:
      pandas table
    """
    prim_keys = ["SID", "lon", "lat"]

    # For T2m (and MAXT2m etc.) we want station elevation for height correction
    if (
        "T2m" in param["harp_param"]
        and "elev" in station_list.columns.to_numpy().tolist()
    ):
        prim_keys.append("elev")
    data = station_list[prim_keys].copy()
    # NOTE: only deterministic mnodels for now! No EPS.
    data[model_name] = data_vector

    # prepare SQLITE output
    fcd = int(fcdate.timestamp())
    vad = int(fcdate.timestamp() + leadtime)

    data["fcst_dttm"] = fcd
    # NOTE: currently FCTABLE still expects leadtime in hours!
    data["lead_time"] = leadtime / 3600.0
    data["valid_dttm"] = vad
    data["parameter"] = param["harp_param"]
    data["units"] = param["units"]
    logger.debug("SQLITE: writing parameter %s", param["harp_param"])
    # model_elevation only relevant for T2m
    # NOTE: we can not yet read model elevation from clim file!
    # otherwise remove column
    if "T2m" in param["harp_param"]:
        if "model_elevation" not in data.columns.to_numpy().tolist():
            data["model_elevation"] = 0
        if "elev" not in data.columns.to_numpy().tolist():
            data["elev"] = 0
    else:
        if "model_elevation" in data.columns.to_numpy().tolist():
            data = data.drop("model_elevation", axis=1)
        if "elev" in data.columns.to_numpy().tolist():
            data = data.drop("elev", axis=1)

    if param["level"] is not None and param["level_name"] is not None:  #  "level"]):
        data[param["level_name"]] = int(param["level"])
    return data


def write_to_sqlite(data, sqlite_file, param, model_name):
    """Write a data table to SQLite.

    Args:
        data: a data table
        sqlite_file: file name
        param: parameter descriptor
        model_name: model name used in file path and data table
    """
    logger.debug("Writing to %s", sqlite_file)

    if os.path.isfile(sqlite_file):
        con = sqlite3.connect(sqlite_file)
        # NOTE: If the table already exists,
        #       we must check that all column names match!
        #       Especially, we must check that the model name is identical.
        #       Otherwise, the FC table should probably be deleted.
        cur = con.cursor()
        cn1 = cur.execute("select name from PRAGMA_TABLE_INFO('FC')").fetchall()
        colnames = [x[0] for x in cn1]
        if model_name not in colnames:
            logger.error(
                "ERROR: The FC table already exists with a different model name!"
            )
            con.close()
            con = None
    else:
        sqlite_path = os.path.dirname(sqlite_file)
        if not os.path.isdir(sqlite_path):
            logger.info("SQLITE: Creating directory %s.", sqlite_path)
            os.makedirs(sqlite_path)
        # if the SQLite file doesn't exist yet: create the SQLite table
        logger.info("SQLITE: Creating sqlite file %s.", sqlite_file)
        con = db_create(sqlite_file, param, model_name)

    if con is not None:
        # NOTE: we need to cast to int (or float),
        #       because "numpy.int64" will not work in SQL
        #       for now, we take float, so the values can be sub-hourly
        #       BUT: this may change if harpPoint gets support for sub-hourly data
        fcd = float(data.iloc[0]["fcst_dttm"])
        leadtime = float(data.iloc[0]["lead_time"])
        logger.debug("leadtime (h): %i", leadtime)
        db_cleanup(param, fcd, leadtime, con)

        # now write to SQLite
        data.to_sql("FC", con, if_exists="append", index=False)
        con.commit()

    con.close()


def db_cleanup(param, fcd, leadtime, con):
    """Delete any prior version of the date (it's a primary key).

    Args:
        param: parameter description
        fcd: forecast date (string or int)
        leadtime: lead time (int)
        con: database connection
    """
    logger.debug("Cleanup: ldt=%i", leadtime)
    cleanup = "DELETE from FC WHERE fcst_dttm=? AND lead_time=?"
    cur = con.cursor()
    # NOTE: variables must be cast to int (or float),
    # because "numpy.int64" will not work in SQL
    # FIXME: better look whether the table has a "level" column!

    cn1 = cur.execute("select name from PRAGMA_TABLE_INFO('FC')").fetchall()
    colnames = [x[0] for x in cn1]

    if param["level_name"] in colnames:
        lname = param["level_name"]
        parlev = int(param["level"])
        cleanup = cleanup + f" AND {lname}=?"
        cur.execute(
            cleanup,
            (
                float(fcd),
                float(leadtime),
                int(parlev),
            ),
        )
    else:
        cur.execute(
            cleanup,
            (
                float(fcd),
                float(leadtime),
            ),
        )
    con.commit()


def db_create(sqlite_file, param, model_name):
    """Create new SQLite file with FC table.

    The table is designed to be compatible with harp, so uses the same
    "unique index" approach.

    Args:
      sqlite_file: SQLite file name
      param: parameter descriptor
      model_name: model name used in SQLite

    Returns:
      data base connection
    """
    primary_keys, all_keys = fctable_definition(param, model_name)
    fc_def = (
        "CREATE table if not exists FC ("
        + ",".join(f"{p[0]} {p[1]} " for p in all_keys.items())
        + ")"
    )
    pk_def = (
        "CREATE unique INDEX IF NOT EXISTS "
        + "index_"
        + "_".join(primary_keys.keys())
        + " ON FC("
        + ",".join(primary_keys.keys())
        + ")"
    )

    con = sqlite3.connect(sqlite_file)
    with con, closing(con.cursor()) as cur:
        cur.execute(fc_def)
        cur.execute(pk_def)
    return con


def fctable_definition(param, model_name):
    """Create the SQL command for FCtable definition.

    Args:
        param: the parameter descriptor
        model_name: model name to be used

    Returns:
        SQL commands for creating the FCTABLE table
    """
    primary_keys = {"fcst_dttm": "DOUBLE", "lead_time": "DOUBLE", "SID": "INT"}
    # if there is a vertical level column, that is also a primary key

    if param["level_name"] in ["z", "p", "h"]:
        primary_keys[param["level_name"]] = "INT"

    other_keys = {
        "lat": "DOUBLE",
        "lon": "DOUBLE",
        "valid_dttm": "INT",
        "parameter": "TEXT",
        "units": "TEXT",
        model_name: "DOUBLE",
    }
    # for T2m correction
    if "T2m" in param["harp_param"]:
        other_keys["elev"] = "DOUBLE"
        other_keys["model_elevation"] = "DOUBLE"

    all_keys = {**primary_keys, **other_keys}
    return (primary_keys, all_keys)
