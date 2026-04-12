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
Air quality plotting functions

Author: Alejandro Rodríguez Sánchez
Contact: ars.rodriguezs@gmail.com
"""

# ruff: noqa

def annual_meteogram_with_pollutant(
    df: pd.DataFrame,
    df_climate: pd.DataFrame,
    year_to_plot: int,
    pollutant: str,
    climate_normal_period: list[int],
    database: str,
    station_name: str,
    filename: str,
    plot_anoms=False,
    show_seasons=True,
    pol_subplot=True
):
    """
    This function allows to plot and compare the annual meteogram of a certain year (year_to_plot) with the climatological normal

    Arguments
    ----------
    df: DataFrame
        DataFrame containing the data
    df_climate: DataFrame
        DataFrame containing the climatological normal data
    year_to_plot: int
        Integer indicating the year to be plotted
    pollutant: int
        String indicating the name of the pollutant to be plotted
    climate_normal_period: list
        List containing the first and last year of the reference period to be used as climatological normal
    database: str
        String representing the name of the database from which the plotted data comes
    station_name: str
        String representing the name of the location of the data
    filename: str
        String containing the absolute path where the plot is going to be saved
    plot_anoms: boolean
        If true, temperature and wind values will be plotted as departures from the climatological selected statistic
    show_seasons: boolean
        If true, the background is plotted with different colors for each climatological season
    pol_subplot: boolean
        If true, the pollutant will be plotted in a separate subplot. If False, it will be plotted in each one of the existing subplots. 
        The latter is only possible if plot_anoms is set to False.

    """

    if "Temp" not in df.columns:
        raise ValueError(
            'The temperature data should be included in df with column name "Temp"'
        )
    if "Rainfall" not in df.columns:
        raise ValueError(
            'The rainfall data should be included in df with column name "Rainfall"'
        )
    if "WindSpeed" not in df.columns:
        raise ValueError(
            'The wind speed data should be included in df with column name "WindSpeed"'
        )

    if pollutant not in df.columns:
        raise ValueError(
            'The pollutant data should be included in df'
        )

    df = df.copy()
    df_climate = df_climate.copy()

    yearmin = df.index.year.min()
    yearmax = df.index.year.max()

    df = df[df.index.year == year_to_plot]
    df_climate = df_climate[df_climate.index.year == year_to_plot]

    df["Accum. Rainfall"] = df["Rainfall"].cumsum()
    df_climate["Accum. Rainfall"] = df_climate["Rainfall_mean"].cumsum()

    # For plotting visuals' purpose
    if df.index.month[-1] != 12 or df.index.month[-1] != 31:
        df = df.reindex(
            pd.date_range(
                "%i-01-01" % year_to_plot, "%i-12-31" % year_to_plot, freq="1D"
            )
        )

    if df_climate.index.month[-1] != 12 or df_climate.index.month[-1] != 31:
        df_climate = df_climate.reindex(
            pd.date_range(
                "%i-01-01" % year_to_plot, "%i-12-31" % year_to_plot, freq="1D"
            )
        )

    # Plotting
    if pol_subplot == True:
        fig, axs = plt.subplots(4, 1, figsize=(15, 17), sharex=True)
        ax = axs.flatten()
        fmt_minor = mdates.DayLocator(interval=1)
        locator = mdates.AutoDateLocator(minticks=10, maxticks=16)
        formatter = mdates.ConciseDateFormatter(
            locator,
            zero_formats=["%d", "%d", "%d", "", "", ""],
            offset_formats=["", "", "", "", "", ""],
        )

        # Pollutant
        (polmedian,) = ax[3].plot(
        df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'].index,
        df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'],
        color="k",
        label="Climate normal",
        )
        if plot_anoms is False:
            ax[3].plot(
                df.loc[df.index.year == year_to_plot, pollutant].index,
                df.loc[df.index.year == year_to_plot, pollutant],
                color="r",
                label="%i" % year_to_plot,
            )
        else:
            diff_var = (
                df.loc[df.index.year == year_to_plot, pollutant]
                - df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median']
            )
            mask1 = diff_var < 0
            mask2 = diff_var >= 0

            if len(mask1[mask1 == True]) > 0:
                ax[3].bar(
                    df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'].index[
                        mask1
                    ],
                    bottom=df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'][
                        mask1
                    ],
                    height=diff_var[mask1],
                    color="#34b1eb",
                    alpha=0.7,
                )
            if len(mask2[mask2 == True]) > 0:
                ax[3].bar(
                    df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'].index[
                        mask2
                    ],
                    bottom=df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'][
                        mask2
                    ],
                    height=diff_var[mask2],
                    color="#eb4034",
                    alpha=0.7,
                )


        # Temperature
        (Tmedian,) = ax[0].plot(
            df_climate.loc[df_climate.index.year == year_to_plot, "Temp"].index,
            df_climate.loc[df_climate.index.year == year_to_plot, "Temp"],
            color="k",
            label="Climate normal",
        )
        if plot_anoms is False:
            ax[0].plot(
                df.loc[df.index.year == year_to_plot, "Temp"].index,
                df.loc[df.index.year == year_to_plot, "Temp"],
                color="r",
                label="%i" % year_to_plot,
            )
        else:
            diff_var = (
                df.loc[df.index.year == year_to_plot, "Temp"]
                - df_climate.loc[df_climate.index.year == year_to_plot, "Temp"]
            )
            mask1 = diff_var < 0
            mask2 = diff_var >= 0

            if len(mask1[mask1 == True]) > 0:
                ax[0].bar(
                    df_climate.loc[df_climate.index.year == year_to_plot, "Temp"].index[
                        mask1
                    ],
                    bottom=df_climate.loc[df_climate.index.year == year_to_plot, "Temp"][
                        mask1
                    ],
                    height=diff_var[mask1],
                    color="#34b1eb",
                    alpha=0.7,
                )
            if len(mask2[mask2 == True]) > 0:
                ax[0].bar(
                    df_climate.loc[df_climate.index.year == year_to_plot, "Temp"].index[
                        mask2
                    ],
                    bottom=df_climate.loc[df_climate.index.year == year_to_plot, "Temp"][
                        mask2
                    ],
                    height=diff_var[mask2],
                    color="#eb4034",
                    alpha=0.7,
                )

        # Rainfall
        (accumrainmedian,) = ax[1].plot(
            df_climate.loc[df_climate.index.year == year_to_plot, "Accum. Rainfall"].index,
            df_climate.loc[df_climate.index.year == year_to_plot, "Accum. Rainfall"],
            color="k",
            label="_nolegend_",
        )
        ax[1].plot(
            df.loc[df.index.year == year_to_plot, "Accum. Rainfall"].index,
            df.loc[df.index.year == year_to_plot, "Accum. Rainfall"],
            color="r",
            label="_nolegend_",
        )
        ax1 = ax[1].twinx()
        ax1.bar(
            x=df_climate.loc[df_climate.index.year == year_to_plot, "Rainfall_mean"].index,
            height=df_climate.loc[df_climate.index.year == year_to_plot, "Rainfall_mean"],
            color="k",
            label="Climate normal",
            alpha=0.7,
        )
        ax1.bar(
            x=df.loc[df.index.year == year_to_plot, "Rainfall"].index,
            height=df.loc[df.index.year == year_to_plot, "Rainfall"],
            color="r",
            label="%i" % year_to_plot,
            alpha=0.7,
        )

        # Wind speed
        (windmedian,) = ax[2].plot(
            df_climate.loc[df_climate.index.year == year_to_plot, "WindSpeed_median"].index,
            df_climate.loc[df_climate.index.year == year_to_plot, "WindSpeed_median"],
            color="k",
            label="Climate normal",
        )
        if plot_anoms is False:
            ax[2].plot(
                df.loc[df.index.year == year_to_plot, "WindSpeed"].index,
                df.loc[df.index.year == year_to_plot, "WindSpeed"],
                color="r",
                label="%i" % year_to_plot,
            )
        else:
            diff_var = (
                df.loc[df.index.year == year_to_plot, "WindSpeed"]
                - df_climate.loc[df_climate.index.year == year_to_plot, "WindSpeed_median"]
            )
            mask1 = diff_var < 0
            mask2 = diff_var >= 0

            if len(mask1[mask1 == True]) > 0:
                ax[2].bar(
                    df_climate.loc[
                        df_climate.index.year == year_to_plot, "WindSpeed"
                    ].index[mask1],
                    bottom=df_climate.loc[
                        df_climate.index.year == year_to_plot, "WindSpeed_median"
                    ][mask1],
                    height=diff_var[mask1],
                    color="#34b1eb",
                    alpha=0.7,
                )
            if len(mask2[mask2 == True]) > 0:
                ax[2].bar(
                    df_climate.loc[
                        df_climate.index.year == year_to_plot, "WindSpeed"
                    ].index[mask2],
                    bottom=df_climate.loc[
                        df_climate.index.year == year_to_plot, "WindSpeed_median"
                    ][mask2],
                    height=diff_var[mask2],
                    color="#eb4034",
                    alpha=0.7,
                )

        for i in range(len(ax)):
            ax[i].tick_params(labelsize=16)
            ax[i].grid(color="black")
            ax[i].legend(fontsize=14)
            ax[i].set_xlim(
                df.index.min() - dt.timedelta(days=2), df.index.max() + dt.timedelta(days=2)
            )
            if i < len(ax) - 1:
                ax[i].set_xticklabels("", fontsize=0)
            else:
                ax[i].xaxis.set_major_locator(locator)
                ax[i].xaxis.set_major_formatter(formatter)

            ax[1].legend().set_visible(False)

            # Show seasons
            if show_seasons is True:
                season_colors = ["#4696db", "#32a852", "#da5757", "#d6db46", "#4696db"]
                ax[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 1, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot), 3, 1)),
                    color="#4696db",
                    alpha=0.2,
                    zorder=-10,
                )
                ax[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 3, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot), 6, 1)),
                    color="#32a852",
                    alpha=0.2,
                    zorder=-10,
                )
                ax[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 6, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot), 9, 1)),
                    color="#da5757",
                    alpha=0.2,
                    zorder=-10,
                )
                ax[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 9, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot), 12, 1)),
                    color="#d6db46",
                    alpha=0.2,
                    zorder=-10,
                )
                ax[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 12, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot) + 1, 1, 1)),
                    color="#4696db",
                    alpha=0.2,
                    zorder=-10,
                )

        ax1.legend(fontsize=14.5)
        ax1.tick_params(labelsize=18)

        ax[0].set_title("Daily temperature", fontsize=21)
        ax[1].set_title("Daily and accumulated rainfall", fontsize=21)
        ax[2].set_title("Daily mean wind speed", fontsize=21)
        ax[3].set_title("Daily mean %s" %pollutant, fontsize=21)

        # Add text with anomalies
        temp_anom = df[df.index == df['Temp'].last_valid_index()]['Temp'] - df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]['Temp']
        
        rain_anom = df[df.index == df['Temp'].last_valid_index()]["Accum. Rainfall"] - df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]["Accum. Rainfall"]
        
        rain_rel_anom = 100 * (df[df.index == df['Temp'].last_valid_index()]["Accum. Rainfall"] - df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]["Accum. Rainfall"]) / df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]["Accum. Rainfall"] 

        wind_anom = df[df.index == df['Temp'].last_valid_index()]["WindSpeed"] - df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]["WindSpeed"]
        
        #bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(        
            0.031,
            0.94,
            str(year_to_plot) + " last temperature anomaly: " + "%+g"
            % (float(temp_anom.values[0])),
            fontsize=15,
            transform=ax[0].transAxes,
            bbox=props
        )

        plt.text(        
            0.031,
            0.94,
            str(year_to_plot) + " accumulated rainfall anomaly: " + "%+g [%+g %%]"
            % (float(rain_anom.values[0]), float(rain_rel_anom.values[0]) ),
            fontsize=15,
            transform=ax[1].transAxes,
            bbox=props
        )

        plt.text(        
            0.031,
            0.94,
            str(year_to_plot) + " last wind anomaly: %+g"
            % (float(wind_anom.values[0])),
            fontsize=15,
            transform=ax[2].transAxes,
            bbox=props
        )    


        text = AnchoredText(
            "Alejandro Rodríguez Sánchez",
            loc=1,
            bbox_to_anchor=(0.24, 0.185),
            bbox_transform=ax[2].transAxes,
            prop={"size": 12},
            frameon=True,
        )
        text.patch.set_alpha(0.5)
        plt.text(
            0.031,
            0.955,
            "Climate normal period: %i-%i"
            % (climate_normal_period[0], climate_normal_period[1]),
            fontsize=16,
            transform=plt.gcf().transFigure,
        )
        plt.text(
            0.031,
            0.93,
            "Period with data: %i-%i" % (yearmin, yearmax),
            fontsize=16,
            transform=plt.gcf().transFigure,
        )
        plt.text(
            0.79,
            0.955,
            "Database: %s" % database,
            fontsize=16,
            transform=plt.gcf().transFigure,
            wrap=True,
        )
        plt.text(
            0.79,
            0.93,
            "Location: %s" % station_name,
            fontsize=16,
            transform=plt.gcf().transFigure,
            wrap=True,
        )
        plt.subplots_adjust(
            left=0.05, right=0.95, hspace=0.12, wspace=0.1, bottom=0.03, top=0.9
        )
        # fig.autofmt_xdate()
        fig.savefig(filename, dpi=300)

    else:
        if plot_anoms == True:
            raise ValueError('"plot_anoms" must be set to False when "pol_subplot" is set to True.')
        
        fig = plt.figure(figsize=(15, 17))
        gs = matplotlib.gridspec.GridSpec(3, 1, hspace=0.3, wspace=0.2, figure=fig)
        fmt_minor = mdates.DayLocator(interval=1)
        locator = mdates.AutoDateLocator(minticks=10, maxticks=16)
        formatter = mdates.ConciseDateFormatter(
            locator,
            zero_formats=["%d", "%d", "%d", "", "", ""],
            offset_formats=["", "", "", "", "", ""],
        )

        # Temperature and pollutant
        gss = matplotlib.gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[0], hspace=0.03
        )
        ax00 = fig.add_subplot(gss[0])
        ax01 = fig.add_subplot(gss[1], sharex=ax00)

        (Tmedian,) = ax00.plot(
            df_climate.loc[df_climate.index.year == year_to_plot, "Temp"].index,
            df_climate.loc[df_climate.index.year == year_to_plot, "Temp"],
            color="k",
            label="Climate normal",
        )
        ax00.plot(
            df.loc[df.index.year == year_to_plot, "Temp"].index,
            df.loc[df.index.year == year_to_plot, "Temp"],
            color="r",
            label="%i" %year_to_plot,
        )
        (polmedian,) = ax01.plot(
        df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'].index,
        df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'],
        color="k",
        ls='--',
        label="Climate normal",
        )
        ax01.plot(
            df.loc[df.index.year == year_to_plot, pollutant].index,
            df.loc[df.index.year == year_to_plot, pollutant],
            color="r",
            ls='--',
            label="%i" % year_to_plot,
        )
        # Rainfall and pollutant
        gss = matplotlib.gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[1], hspace=0.03
        )
        ax10 = fig.add_subplot(gss[0])
        ax11 = fig.add_subplot(gss[1], sharex=ax10)

        ax10.bar(
            x=df_climate.loc[df_climate.index.year == year_to_plot, "Rainfall_mean"].index,
            height=df_climate.loc[df_climate.index.year == year_to_plot, "Rainfall_mean"],
            color="k",
            label="Climate normal",
            alpha=0.7,
        )
        ax10.bar(
            x=df.loc[df.index.year == year_to_plot, "Rainfall"].index,
            height=df.loc[df.index.year == year_to_plot, "Rainfall"],
            color="r",
            label="%i" % year_to_plot,
            alpha=0.7,
        )


        (polmedian,) = ax11.plot(
        df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'].index,
        df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'],
        color="k",
        ls='--',
        label="Climate normal",
        )
        ax11.plot(
            df.loc[df.index.year == year_to_plot, pollutant].index,
            df.loc[df.index.year == year_to_plot, pollutant],
            color="r",
            ls='--',
            label="%i" % year_to_plot,
        )


        # Wind speed and pollutant
        gss = matplotlib.gridspec.GridSpecFromSubplotSpec(
            2, 1, subplot_spec=gs[2], hspace=0.03
        )
        ax20 = fig.add_subplot(gss[0])
        ax21 = fig.add_subplot(gss[1], sharex=ax20)

        (windmedian,) = ax20.plot(
            df_climate.loc[df_climate.index.year == year_to_plot, "WindSpeed_median"].index,
            df_climate.loc[df_climate.index.year == year_to_plot, "WindSpeed_median"],
            color="k",
            label="Climate normal",
        )
        ax20.plot(
            df.loc[df.index.year == year_to_plot, "WindSpeed"].index,
            df.loc[df.index.year == year_to_plot, "WindSpeed"],
            color="r",
            label="%i" % year_to_plot,
        )

        (polmedian,) = ax21.plot(
        df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'].index,
        df_climate.loc[df_climate.index.year == year_to_plot, pollutant+'_median'],
        color="k",
        ls='--',
        label="Climate normal",
        )
        ax21.plot(
            df.loc[df.index.year == year_to_plot, pollutant].index,
            df.loc[df.index.year == year_to_plot, pollutant],
            color="r",
            ls='--',
            label="%i" % year_to_plot,
        )

        axes = [ax00, ax01, ax10, ax11, ax20, ax21]


        for i in range(len(axes)):
            axes[i].tick_params(labelsize=16)
            axes[i].grid(color="black")
            axes[i].legend(fontsize=14)
            axes[i].set_xlim(
                df.index.min() - dt.timedelta(days=2), df.index.max() + dt.timedelta(days=2)
            )
            if i < len(axes) - 1:
                axes[i].set_xticklabels("", fontsize=0)
            else:
                axes[i].xaxis.set_major_locator(locator)
                axes[i].xaxis.set_major_formatter(formatter)

            #ax[1].legend().set_visible(False)

            # Show seasons
            if show_seasons is True:
                season_colors = ["#4696db", "#32a852", "#da5757", "#d6db46", "#4696db"]
                axes[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 1, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot), 3, 1)),
                    color="#4696db",
                    alpha=0.2,
                    zorder=-10,
                )
                axes[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 3, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot), 6, 1)),
                    color="#32a852",
                    alpha=0.2,
                    zorder=-10,
                )
                axes[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 6, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot), 9, 1)),
                    color="#da5757",
                    alpha=0.2,
                    zorder=-10,
                )
                axes[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 9, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot), 12, 1)),
                    color="#d6db46",
                    alpha=0.2,
                    zorder=-10,
                )
                axes[i].axvspan(
                    mdates.date2num(dt.datetime(int(year_to_plot), 12, 1)),
                    mdates.date2num(dt.datetime(int(year_to_plot) + 1, 1, 1)),
                    color="#4696db",
                    alpha=0.2,
                    zorder=-10,
                )



        ax00.set_title("Daily temperature and %s" %pollutant, fontsize=21)
        ax10.set_title("Daily and accumulated rainfall and %s" %pollutant, fontsize=21)
        ax20.set_title("Daily mean wind speed and %s" %pollutant, fontsize=21)

        # Add text with anomalies
        temp_anom = df[df.index == df['Temp'].last_valid_index()]['Temp'] - df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]['Temp']
        
        rain_anom = df[df.index == df['Temp'].last_valid_index()]["Accum. Rainfall"] - df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]["Accum. Rainfall"]
        
        rain_rel_anom = 100 * (df[df.index == df['Temp'].last_valid_index()]["Accum. Rainfall"] - df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]["Accum. Rainfall"]) / df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]["Accum. Rainfall"] 

        wind_anom = df[df.index == df['Temp'].last_valid_index()]["WindSpeed"] - df_climate[
            (df_climate.index.month == df['Temp'].last_valid_index().month) & 
            (df_climate.index.day == df['Temp'].last_valid_index().day)]["WindSpeed"]
        
        #bbox = dict(boxstyle='round', fc='blanchedalmond', ec='orange', alpha=0.5)
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        plt.text(        
            0.031,
            0.94,
            str(year_to_plot) + " last temperature anomaly: " + "%+g"
            % (float(temp_anom.values[0])),
            fontsize=15,
            transform=ax00.transAxes,
            bbox=props
        )

        plt.text(        
            0.031,
            0.94,
            str(year_to_plot) + " accumulated rainfall anomaly: " + "%+g [%+g %%]"
            % (float(rain_anom.values[0]), float(rain_rel_anom.values[0]) ),
            fontsize=15,
            transform=ax10.transAxes,
            bbox=props
        )

        plt.text(        
            0.031,
            0.94,
            str(year_to_plot) + " last wind anomaly: %+g"
            % (float(wind_anom.values[0])),
            fontsize=15,
            transform=ax20.transAxes,
            bbox=props
        )    


        text = AnchoredText(
            "Alejandro Rodríguez Sánchez",
            loc=1,
            bbox_to_anchor=(0.24, 0.185),
            bbox_transform=ax20.transAxes,
            prop={"size": 12},
            frameon=True,
        )
        text.patch.set_alpha(0.5)
        plt.text(
            0.031,
            0.955,
            "Climate normal period: %i-%i"
            % (climate_normal_period[0], climate_normal_period[1]),
            fontsize=16,
            transform=plt.gcf().transFigure,
        )
        plt.text(
            0.031,
            0.93,
            "Period with data: %i-%i" % (yearmin, yearmax),
            fontsize=16,
            transform=plt.gcf().transFigure,
        )
        plt.text(
            0.79,
            0.955,
            "Database: %s" % database,
            fontsize=16,
            transform=plt.gcf().transFigure,
            wrap=True,
        )
        plt.text(
            0.79,
            0.93,
            "Location: %s" % station_name,
            fontsize=16,
            transform=plt.gcf().transFigure,
            wrap=True,
        )
        plt.subplots_adjust(
            left=0.05, right=0.95, hspace=0.12, wspace=0.1, bottom=0.03, top=0.9
        )
        # fig.autofmt_xdate()
        fig.savefig(filename, dpi=300)