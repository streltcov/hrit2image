"""
hrit2image

"""


from glob import glob

import numpy as np
import pandas as pd
import click
import matplotlib.pyplot as plt
from cartopy.crs import Mercator, PlateCarree

from satpy import Scene
from pyresample.geometry import SwathDefinition


########################################################################################################################

CHANNELS = {
    "vis006": "VIS006",
    "wv062": "WV062",
    "wv073": "WV073",
    "ir108": "IR_108",
    "hrv": "HRV",
}

PROJECTIONS = {
    'mercator': Mercator,
    'platecarree': PlateCarree 
}

########################################################################################################################

CHANNEL_HELP = "One of SEVIRI infrared channels; available values: vis006, wv062, wv073, ir108, hrv"
PROJECTION_HELP = "One of Cartopy projections; available values: mercator, platecarree, geostationary, orthographic, mollweide"
TOLERANCE_HELP = ""

########################################################################################################################

#******************#
#  Data functions  #
#******************#


def __convert_coordinate(value):
    """

    Args:
        value (str): a coordinate value in string format

    Returns:
        float: converted coordinate value
    """
    value, modifier = str(value).split('.')[0], ''
    if value[:1] == '-':
        value, modifier = value[1:], '-'
    integer_part, float_part = value[:2], value[3:]
    return float(modifier + integer_part + '.' + float_part)


def __define_swath_area(scene, channel):
    """ """
    lons_row = np.array(list(map(__convert_coordinate, [float(x) for x in scene[channel]['x']])))
    lats_row = np.array(list(map(__convert_coordinate, [float(y) for y in scene[channel]['y']])))
    definition = SwathDefinition(lons=lons_row, lats=lats_row, nprocs=2)
    return definition.lons, definition.lats


def __create_levels(df):
    """ """
    levels = {}
    levels[1] = df.query('value > 330')
    levels[2] = df.query('value <= 330 and value > 320')
    levels[3] = df.query('value <= 320 and value > 310')
    levels[4] = df.query('value <= 310 and value > 300')
    levels[5] = df.query('value <= 300 and value > 295')
    levels[6] = df.query('value <= 295 and value > 290')
    levels[7] = df.query('value <= 290 and value > 285')
    levels[8] = df.query('value <= 285 and value > 280')
    levels[9] = df.query('value <= 280 and value > 270')
    levels[10] = df.query('value <= 270 and value > 260')
    levels[11] = df.query('value <= 260 and value > 250')
    levels[12] = df.query('value <= 250 and value > 240')
    levels[13] = df.query('value <= 240 and value > 230')
    levels[14] = df.query('value <= 230 and value > 220')
    levels[15] = df.query('value <= 220 and value > 210')
    levels[16] = df.query('value <= 210 and value > 200')
    levels[17] = df.query('value <= 200 and value > 180')
    levels[18] = df.query('value <=180')
    return levels

########################################################################################################################

#****************#
#  CLI commands  #
#****************#


@click.group()
def cli():
    """ """


@click.command()
@click.argument('path')
def msg_info(path):
    """ """
    filenames = glob(path + '*')
    scn = Scene(reader='seviri_l1b_hrit', filenames=filenames)


@click.command()
@click.argument("path")
@click.option("--channel", "-c",  help=CHANNEL_HELP)
@click.option("--projection", "-p", help=PROJECTION_HELP)
@click.option("--tolerance", "-t", help=TOLERANCE_HELP)
def msg2cartopy(path, channel="IR_108", projection="mercator", tolerance=1.0):
    """ Plot MSG HRIT file via Cartopy library """

    channel = CHANNELS.get(channel, "IR_108")
    filenames = glob(path + "*")

    scn = Scene(reader="seviri_l1b_hrit", filenames=filenames)
    scn.load([channel])

    values = np.array(scn[channel].values)

    # lons, lats = __define_swath_area(scn, channel)
    lons = np.array([list(map(__convert_coordinate, [float(x) for x in scn[channel]['x']])), ] * 3712).reshape(3712 **2)
    lats = np.array([list(map(__convert_coordinate, [float(y) for y in scn[channel]['y']])), ] * 3712).T.reshape(3712 ** 2)

    values = list(values.reshape(3712 ** 2))

    df = pd.DataFrame.from_records([{'value': values, 'latitude': lats, 'longitude': lons}])
    print(df.columns)
    print(df.head())
    # :levels = __create_levels(df)


########################################################################################################################

cli.add_command(msg_info)
cli.add_command(msg2cartopy)

########################################################################################################################

if __name__ == "__main__":
    cli()

