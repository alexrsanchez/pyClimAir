import numpy as np
import pandas as pd
import datetime as dt
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.offsetbox import AnchoredText
import warnings
from scipy import stats, interpolate
import os
from pathlib import Path

"""
Plotting functions for climatological data

Author: Alejandro Rodríguez Sánchez
Contact: ars.rodriguezs@gmail.com
"""

# ruff: noqa

def compare_with_globaldataset(
    df: pd.DataFrame,
    var: str,
    units: str,
    database: str,
    station_name: str,
    filename: str,
    window=None,
    global_dataset="HadCRUT",
):
    """
    This function allows to compare the annual trend of your timeseries with that of one global dataset

    Parameters
    ----------
    df: DataFrame
        DataFrame containing the data
    var: str
        String containing the name of the variable to be plotted
    units: str
        String containing the units of the variable to be plotted
    database: str
        String containing the name of the database from which the plotted data comes
    station_name: str
        String containing the name of the location of the data
    filename: str
        String containing the absolute path where the plot is going to be saved
    window: int
        Integer representing the length of the window for moving average computation. If None, no moving average is computed and the original data anomalies are plotted.
    global_dataset: str
        String containing the name of the dataset to compare with. At the moment, only "HadCRUT" is accepted.
    """

    df = df.copy()

    yearmin = df[var].first_valid_index().year  # df.index.year.min()
    yearmax = df[var].last_valid_index().year

    path = Path(os.path.dirname(__file__))  # Current directory

    normal_values = df[(df.index.year >= 1961) & (df.index.year <= 1990)]
    if len(normal_values) == 0:
        print(
            "Your data has not data within the period 1961-1990. Therefore, there cannot be perform a comparison with the global dataset. Exiting..."
        )
        return

    normal_values = (
        normal_values.groupby(normal_values.index)
        .mean(numeric_only=True)[var]
        .mean(numeric_only=True)
    )

    df_plot = df.groupby(df.index.year).mean(numeric_only=True) - float(normal_values)
    label = station_name
    title = "%s anomaly [%s]" % (var, units)

    if global_dataset == "HadCRUT":
        df1 = pd.read_csv(
            str(path.parent.absolute())
            + "\data\HadCRUT.5.1.0.0\HadCRUT.5.1.0.0.analysis.summary_series.global.annual.csv",
            sep=",",
            decimal=".",
            header=0,
        )

    if window is not None:
        df_plot = df_plot.rolling(window=window).mean()
        df1 = df1.rolling(window=window).mean()
        title = "%s anomaly [%s] %i-year moving average" % (var, units, window)

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(df1.iloc[:, 0], df1.iloc[:, 1], color="k", lw=2, label=global_dataset)
    ax.fill_between(
        df1.iloc[:, 0],
        df1.iloc[:, 2],
        df1.iloc[:, 3],
        color="grey",
        lw=2,
        label="%s 95%% confidence interval" % global_dataset,
        alpha=0.7,
    )
    ax.plot(df_plot[var], color="blue", lw=1.5, label=label)
    ax.grid(color="k", linestyle="--")
    ax.axhline(y=0, color="k")
    ax.tick_params(axis="both", labelsize=16)
    ax.set_ylabel("Anomaly to 1961-1900 [ºC]", fontsize=16)

    plt.text(
        0.03,
        0.96,
        "Climate normal period: 1961-1990",
        fontsize=12,
        transform=plt.gcf().transFigure,
    )
    plt.text(
        0.03,
        0.93,
        "Period with data: %i-%i" % (yearmin, yearmax),
        fontsize=12,
        transform=plt.gcf().transFigure,
    )
    plt.text(
        0.77,
        0.96,
        "Database: %s" % database,
        fontsize=12,
        transform=plt.gcf().transFigure,
    )
    plt.text(
        0.77,
        0.93,
        "Location: %s" % station_name,
        fontsize=12,
        transform=plt.gcf().transFigure,
    )
    fig.suptitle(title, fontsize=24)
    ax.legend(fontsize=14).set_visible(True)
    ax.margins(0.01, 0.05)
    plt.tight_layout()
    plt.savefig(filename, dpi=300)

