"""
hrit2image

"""


from glob import glob

import numpy as np
import pandas as pd
import click
import matplotlib
import matplotlib.pyplot as plt
from cartopy.crs import (
    PlateCarree,
    AlbersEqualArea,
    AzimuthalEquidistant,
    EquidistantConic,
    LambertConformal,
    Mercator,
    Miller,
    Mollweide,
    Orthographic,
    Robinson,
    Geostationary,
    NearsidePerspective,
    EqualEarth,
    NorthPolarStereo,
    SouthPolarStereo,
)

from satpy import Scene


########################################################################################################################

# Available infrared channels for SEVIRI instrument;
__SEVIRI_CHANNELS = {
    "vis006": "VIS006",
    "wv062": "WV062",
    "wv073": "WV073",
    "ir108": "IR_108",
    "ir120": "IR_120",
    "hrv": "HRV",
}

__CARTOPY_PROJECTIONS = {
    "platecarree": PlateCarree,
    "lambert": LambertConformal,
    "mercator": Mercator,
    "miller": Miller,
    "mollweide": Mollweide,
    "orthographic": Orthographic,
    "robinson": Robinson,
    "geostationary": Geostationary,
    "nearside": NearsidePerspective,
    "equalearth": EqualEarth,
    "northstereo": NorthPolarStereo,
    "southstereo": SouthPolarStereo,
}

__GRAYSCALE = {
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

__ALPHA_LEVELS = {
    1: 0.8,
    2: 0.7,
    3: 0.7,
    4: 0.7,
    5: 0.7,
    6: 0.5,
    7: 0.5,
    8: 0.5,
    9: 0.5,
    10: 0.4,
    11: 0.3,
    12: 0.3,
    13: 0.3,
    14: 0.3,
    15: 0.3,
    16: 0.3,
    17: 0.3,
    18: 0.3,
}

__SIZES = {"l": (100, 100), "m": (150, 150), "b": (200, 200)}

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


def __create_brightness_temparature_levels(df):
    """

    Args:
        df (pandas.DataFrame):
    Returns:
        dict: a dictionary of DataFrame objects;
    """
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


def __create_viirs_projection(
    projection_type: str,
    satellite_height=35800.0,
    central_latitude=0.0,
    central_longitude=0.0,
):
    """Creates a cartopy.crs object;

    Args:
        projection_type (str):
        satellite_height (float):
        central_longitude (float):
        central_latitude (float):

    Returns:
        cartopy.crs.Projection: a Cartopy projection object;
    """

    __longitude_only = [
        "platecarree",
        "miller",
        "mollweide",
        "orthographic",
        "robinson",
        "geostationary",
        "",
    ]

    projection_cls = (
        __CARTOPY_PROJECTIONS[projection_type]
        if projection_type in __CARTOPY_PROJECTIONS
        else PlateCarree
    )

    if projection_type in __longitude_only:
        return projection_cls(central_longitude=central_longitude)

    return projection_cls(
        central_longitude=central_longitude, central_latitude=central_latitude
    )


########################################################################################################################

# ****************#
#  CLI commands  #
# ****************#


@click.group()
def cli():
    """ """


@click.command()
def test_command():
    """ """
    projection = __create_viirs_projection("mollweide")
    print(type(projection))


@click.command()
@click.argument("path")
def msg_scene_info(path):
    """ """
    filenames = glob(path + "*")

    scn = Scene(reader="seviri_l1b_hrit", filenames=filenames)
    scn.load(scn.available_dataset_names())
    dataset = scn.to_xarray_dataset()

    print("Available datasets: " + "".join(scn.available_dataset_names()))
    print("Satellite altitude: " + str(dataset.satellite_altitude))
    print("Satellite longitude: " + str(dataset.satellite_longitude))
    print("Satellite latitude: " + str(dataset.satellite_latitude))
    print("Sensor: " + str(dataset.sensor))

    crs = Mercator()


@click.command()
@click.argument("path")
@click.option("--channel", "-c", help=CHANNEL_HELP)
@click.option("--projection", "-p", help="")
@click.option("--palette", help="")
@click.option("--writer", "-w", help="")
@click.option("--output", "-o", help="")
def msg2img(
    path,
    channel="ir108",
    projection="geostationary",
    palette="normal",
    writer="",
    output="scene",
):
    """ """
    filenames = glob(path + "*")
    scn = Scene(reader="seviri_l1b_hrit", filenames=filenames)
    channel = (
        __SEVIRI_CHANNELS[channel] if channel in __SEVIRI_CHANNELS.keys() else "IR_108"
    )
    scn.load([channel])


@click.command()
@click.argument("path")
@click.option("--output", "-o", default="sample.png", help="Output file name")
@click.option("--channel", "-c", default="ir108", help=CHANNEL_HELP)
@click.option("--projection", "-p", default="platecarree", help=PROJECTION_HELP)
@click.option("--view-longitude", default=None, help="")
@click.option("--view-latitude", default=None, help="")
@click.option("--image-size", "-s", default="m", help="Output image size (s, m or b)")
@click.option(
    "--stock-image", is_flag=True, help="Boolean option - draw background map"
)
@click.option("--coastlines", is_flag=True, help="Boolean option - draw coastlines")
@click.option("--nightshade", is_flag=False, help="")
@click.option("--no-water", is_flag=True, help="")
@click.option("--grid", is_flag=True, help="")
def msg2cartopy(
    path,
    output=None,
    channel="ir108",
    projection="platecarree",
    view_longitude=None,
    view_latitude=None,
    image_size="m",
    stock_image=True,
    coastlines=True,
    nightshade=False,
    no_water=False,
    grid=True,
):
    """ Plot MSG HRIT file via Cartopy library """

    matplotlib.use("Agg")
    plt.ioff()

    channel = __SEVIRI_CHANNELS.get(channel, "IR_108")
    filenames = glob(path + "*")

    scn = Scene(reader="seviri_l1b_hrit", filenames=filenames)
    scn.load([channel])

    figsize = __SIZES[image_size] if image_size in __SIZES.keys() else __SIZES["m"]

    # xarray dataset (priomarily coordinates and brightness temperature values);
    dataset = scn.to_xarray_dataset()

    # satellite position data (altitude, longitude and latitude);
    satellite_altitude = dataset.satellite_altitude
    satellite_longitude = dataset.satellite_longitude
    satellite_latitude = dataset.satellite_latitude

    view_longitude = view_longitude if view_longitude else satellite_longitude
    view_latitude = view_latitude if view_latitude else satellite_latitude

    coords = dataset.coords
    coords_dataset = coords.to_dataset()
    data = coords_dataset.to_dataframe()
    longitudes = data.index.get_level_values("x")
    latitudes = data.index.get_level_values("y")

    # brightness temperature values array;
    values = scn[channel].values
    values = values.reshape(3712 ** 2)

    dataframe = pd.DataFrame(
        {"value": values, "longitude": longitudes, "latitude": latitudes}
    )

    # drop items where brightness value is NaN;
    dataframe = dataframe.dropna()

    # remove water if necessary;
    if no_water:
        dataframe = dataframe.query("value <= 265")

    levels = __create_brightness_temparature_levels(dataframe)

    ####################

    map_projection = __create_viirs_projection(
        projection,
        satellite_height=satellite_altitude,
        central_longitude=view_longitude,
        central_latitude=view_latitude,
    )

    figure = plt.figure(figsize=(100, 100))
    ax = plt.axes(projection=map_projection)

    # creating MSG projection object;
    viirs_projection = Geostationary(central_longitude=satellite_longitude)

    if stock_image:
        ax.stock_img()

    if grid:
        grid_projection_cls = __CARTOPY_PROJECTIONS[projection]
        ax.gridlines(crs=grid_projection_cls(), linestyle="--", color="cyan", zorder=30)

    for number, level in levels.items():
        ax.scatter(
            level.longitude,
            level.latitude,
            color=__GRAYSCALE[number],
            marker=",",
            alpha=__ALPHA_LEVELS[number],
            transform=viirs_projection,
            zorder=number,
        )

    plt.savefig(output)
    quit()


########################################################################################################################

cli.add_command(msg_scene_info)
cli.add_command(msg2img)
cli.add_command(msg2cartopy)
cli.add_command(test_command)

########################################################################################################################

if __name__ == "__main__":
    cli()
