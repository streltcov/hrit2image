"""
hrit2image

"""


import pathlib
from glob import glob

import numpy as np
import pandas as pd
import click
import matplotlib
import matplotlib.pyplot as plt
import geopandas as gpd
from cartopy.crs import Mercator, PlateCarree, Orthographic, Geostationary

from satpy import Scene


########################################################################################################################

SEVIRI_CHANNELS = {
    "vis006": "VIS006",
    "wv062": "WV062",
    "wv073": "WV073",
    "ir108": "IR_108",
    "hrv": "HRV",
}

CARTOPY_PROJECTIONS = {
    "mercator": Mercator,
    "platecarree": PlateCarree,
    "orthographic": Orthographic,
    "geostationary": Geostationary,
}

GRAYSCALE_PALETTE = {
    1: "#202020",
    2: "#282828",
    3: "#303030",
    4: "#383838",
    5: "#404040",
    6: "#484848",
    7: "#505050",
    8: "#585858",
    9: "#686868",
    10: "#787878",
    11: "#888888",
    12: "#989898",
    13: "#A8A8A8",
    14: "#B8B8B8",
    15: "#C8C8C8",
    16: "#D8D8D8",
    17: "#E8E8E8",
    18: "#F8F8F8",
}

########################################################################################################################

CHANNEL_HELP = "One of SEVIRI infrared channels; available values: vis006, wv062, wv073, ir108, hrv"
PROJECTION_HELP = "One of Cartopy projections; available values: mercator, platecarree, geostationary, orthographic, mollweide"
TOLERANCE_HELP = ""

########################################################################################################################

# ******************#
#  Data functions  #
# ******************#


def __convert_string_coordinate(value):
    """

    Args:
        value (str): a coordinate value in string format

    Returns:
        float: converted coordinate value
    """
    value, modifier = str(value).split(".")[0], ""
    if value[:1] == "-":
        value, modifier = value[1:], "-"
    integer_part, float_part = value[:2], value[3:]
    return float(modifier + integer_part + "." + float_part)


def __define_swath_area(scene, channel):
    """ """
    lons_row = np.array(
        list(map(__convert_coordinate, [float(x) for x in scene[channel]["x"]]))
    )
    lats_row = np.array(
        list(map(__convert_coordinate, [float(y) for y in scene[channel]["y"]]))
    )
    definition = SwathDefinition(lons=lons_row, lats=lats_row, nprocs=2)
    return definition.lons, definition.lats


def __create_levels(df):
    """ """
    levels = {}
    levels[1] = df.query("value > 330")
    levels[2] = df.query("value <= 330 and value > 320")
    levels[3] = df.query("value <= 320 and value > 310")
    levels[4] = df.query("value <= 310 and value > 300")
    levels[5] = df.query("value <= 300 and value > 295")
    levels[6] = df.query("value <= 295 and value > 290")
    levels[7] = df.query("value <= 290 and value > 285")
    levels[8] = df.query("value <= 285 and value > 280")
    levels[9] = df.query("value <= 280 and value > 270")
    levels[10] = df.query("value <= 270 and value > 260")
    levels[11] = df.query("value <= 260 and value > 250")
    levels[12] = df.query("value <= 250 and value > 240")
    levels[13] = df.query("value <= 240 and value > 230")
    levels[14] = df.query("value <= 230 and value > 220")
    levels[15] = df.query("value <= 220 and value > 210")
    levels[16] = df.query("value <= 210 and value > 200")
    levels[17] = df.query("value <= 200 and value > 180")
    levels[18] = df.query("value <=180")
    return levels


########################################################################################################################

# ****************#
#  CLI commands  #
# ****************#


@click.group()
def cli():
    """ """


@click.command()
@click.argument("path")
def msg_info(path):
    """ """
    filenames = glob(path + "*")
    scn = Scene(reader="seviri_l1b_hrit", filenames=filenames)


@click.command()
@click.argument('path')
@click.option('--channel', '-c', help=CHANNEL_HELP)
@click.option('--projection', '-p', help='')
@click.option('--palette', help='')
@click.option('--writer', '-w', help='')
@click.option('--output', '-o', help='')
def msg2img(path, channel='ir108', projection='geostationary', palette='normal', writer='',  output='scene'):
    """ """
    filenames = glob(path + '*')
    scn = Scene(reader='seviri_l1b_hrit', filenames=filenames)
    channel = SEVIRI_CHANNELS[channel] if channel in SEVIRI_CHANNELS.keys() else 'IR_108'
    scn.load([channel])

    if projection == 'geostationary':
        scn.save_dataset(channel, filename=output)


@click.command()
@click.argument("path")
@click.option("--channel", "-c", help=CHANNEL_HELP)
@click.option("--projection", "-p", help=PROJECTION_HELP)
@click.option("--tolerance", "-t", help=TOLERANCE_HELP)
def msg2cartopy(path, channel="ir108", projection="mercator", tolerance=1.0):
    """ Plot MSG HRIT file via Cartopy library """

    matplotlib.use("Agg")
    plt.ioff()

    channel = SEVIRI_CHANNELS.get(channel, "IR_108")
    filenames = glob(path + "*")

    scn = Scene(reader="seviri_l1b_hrit", filenames=filenames)
    scn.load([channel])

    dataset = scn.to_xarray_dataset()
    coords = dataset.coords
    coords_dataset = coords.to_dataset()
    data = coords_dataset.to_dataframe()
    longitudes = data.index.get_level_values('x')
    latitudes = data.index.get_level_values('y')

    values = scn[channel].values
    values = values.reshape(3712 ** 2)

    dataframe = pd.DataFrame({'value': values, 'longitude': longitudes, 'latitude': latitudes})
    dataframe = dataframe.dropna()

    levels = __create_levels(dataframe)

    ####################

    figure = plt.figure(figsize=(100, 100))
    ax = plt.axes(projection=PlateCarree())

    viirs_projection = Orthographic()

    ax.stock_img()

    ax.scatter(levels[1].longitude, levels[1].latitude, color=GRAYSCALE_PALETTE[1], marker=',', alpha=0.8, transform=viirs_projection, zorder=1)
    ax.scatter(levels[2].longitude, levels[2].latitude, color=GRAYSCALE_PALETTE[2], marker=',', alpha=0.7, transform=viirs_projection, zorder=2)
    ax.scatter(levels[3].longitude, levels[3].latitude, color=GRAYSCALE_PALETTE[3], marker=',', alpha=0.7, transform=viirs_projection, zorder=3)
    ax.scatter(levels[4].longitude, levels[4].latitude, color=GRAYSCALE_PALETTE[4], marker=',', alpha=0.7, transform=viirs_projection, zorder=4)
    ax.scatter(levels[5].longitude, levels[5].latitude, color=GRAYSCALE_PALETTE[5], marker=',', alpha=0.7, transform=viirs_projection, zorder=5)
    ax.scatter(levels[6].longitude, levels[6].latitude, color=GRAYSCALE_PALETTE[6], marker=',', alpha=0.5, transform=viirs_projection, zorder=6)
    ax.scatter(levels[7].longitude, levels[7].latitude, color=GRAYSCALE_PALETTE[7], marker=',', alpha=0.5, transform=viirs_projection, zorder=7)
    ax.scatter(levels[8].longitude, levels[8].latitude, color=GRAYSCALE_PALETTE[8], marker=',', alpha=0.5, transform=viirs_projection, zorder=8)
    ax.scatter(levels[9].longitude, levels[9].latitude, color=GRAYSCALE_PALETTE[9], marker=',', alpha=0.5, transform=viirs_projection, zorder=9)
    ax.scatter(levels[10].longitude, levels[10].latitude, color=GRAYSCALE_PALETTE[10], marker=',', alpha=0.4, transform=viirs_projection, zorder=10)
    ax.scatter(levels[11].longitude, levels[11].latitude, color=GRAYSCALE_PALETTE[11], marker=',', alpha=0.4, transform=viirs_projection, zorder=11)
    ax.scatter(levels[12].longitude, levels[12].latitude, color=GRAYSCALE_PALETTE[12], marker=',', alpha=0.3, transform=viirs_projection, zorder=12)
    ax.scatter(levels[13].longitude, levels[13].latitude, color=GRAYSCALE_PALETTE[13], marker=',', alpha=0.3, transform=viirs_projection, zorder=13)
    ax.scatter(levels[14].longitude, levels[14].latitude, color=GRAYSCALE_PALETTE[14], marker=',', alpha=0.3, transform=viirs_projection, zorder=14)
    ax.scatter(levels[15].longitude, levels[15].latitude, color=GRAYSCALE_PALETTE[15], marker=',', alpha=0.3, transform=viirs_projection, zorder=15)
    ax.scatter(levels[16].longitude, levels[16].latitude, color=GRAYSCALE_PALETTE[16], marker=',', alpha=0.3, transform=viirs_projection, zorder=16)
    ax.scatter(levels[17].longitude, levels[17].latitude, color=GRAYSCALE_PALETTE[17], marker=',', alpha=0.3, transform=viirs_projection, zorder=17)
    ax.scatter(levels[18].longitude, levels[18].latitude, color=GRAYSCALE_PALETTE[18], marker=',', alpha=0.3, transform=viirs_projection, zorder=18)

    plt.savefig('cartopy_sample.png')


########################################################################################################################

cli.add_command(msg_info)
cli.add_command(msg2img)
cli.add_command(msg2cartopy)

########################################################################################################################

if __name__ == "__main__":
    cli()
