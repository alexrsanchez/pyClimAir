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