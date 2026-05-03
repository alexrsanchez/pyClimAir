import matplotlib
import numpy as np
import matplotlib.pyplot as plt

def fill_between_colormap(x, y1, y2, cmap, **kwargs):
    cmap = matplotlib.cm.get_cmap("RdBu_r")

    xx = x.values
    yy1 = y1.values
    yy2 = y2.values

    yy = yy1 - yy2

    extreme_value = max(abs(np.nanmin(yy)), abs(np.nanmax(yy)))

    normalize = matplotlib.colors.Normalize(vmin=-extreme_value, vmax=extreme_value)
    npts = len(xx)
    for i in range(npts - 1):
        a = plt.fill_between(
            [xx[i], xx[i + 1]],
            [yy1[i], yy1[i + 1]],
            [yy2[i], yy2[i + 1]],
            color=cmap(normalize(yy[i])),
            edgecolor=None,
            **kwargs,
        )

    return cmap, normalize

def hex_to_rgb(value):
    """
    Converts hex to rgb colours
    value: string of 6 characters representing a hex colour.
    Returns: list length 3 of RGB values"""

    ### From: https://github.com/CJP123/continuous_cmap/

    value = value.strip("#")  # removes hash symbol if present
    lv = len(value)
    return tuple(int(value[i : i + lv // 3], 16) for i in range(0, lv, lv // 3))


def rgb_to_dec(value):
    """
    Converts rgb to decimal colours (i.e. divides each value by 256)
    value: list (length 3) of RGB values
    Returns: list (length 3) of decimal values"""

    ### From: https://github.com/CJP123/continuous_cmap/

    return [v / 256 for v in value]


def get_continuous_cmap(hex_list, float_list=None, N_colors=256):
    """creates and returns a color map that can be used in heat map figures.
    If float_list is not provided, colour map graduates linearly between each color in hex_list.
    If float_list is provided, each color in hex_list is mapped to the respective location in float_list.

    Parameters
    ----------
    hex_list: list of hex code strings
    float_list: list of floats between 0 and 1, same length as hex_list. Must start with 0 and end with 1.

    Returns
    ----------
    colour map"""

    ### From: https://github.com/CJP123/continuous_cmap/

    rgb_list = [rgb_to_dec(hex_to_rgb(i)) for i in hex_list]
    if float_list:
        pass
    else:
        float_list = list(np.linspace(0, 1, len(rgb_list)))

    cdict = dict()
    for num, col in enumerate(["red", "green", "blue"]):
        col_list = [
            [float_list[i], rgb_list[i][num], rgb_list[i][num]]
            for i in range(len(float_list))
        ]
        cdict[col] = col_list
    cmp = matplotlib.colors.LinearSegmentedColormap(
        "my_cmp", segmentdata=cdict, N=N_colors
    )
    return cmp
