
# First: Import Python file(s) with all the functions
from ._common import (
    quality_control,
    compute_climate,
    compute_daily_records_oneyear,
    compute_records,
    plot_records_count,
    compute_and_plot_exceedances,
    plot_variable_trends,
    plot_data_vs_climate,
    plot_data_vs_climate_withrecords,
    plot_data_vs_climate_withrecords_multivar,
    plot_periodstats,
    plot_data_and_accum_anoms,
    plot_data_and_annual_cycle,
    plot_timeseries,
    timeseries_extremevalues,
    plot_annual_cycles,
    get_annual_cycle,
    annual_meteogram,
    plot_accumulated_anomalies,
    plot_anomalies,
    compare_probdist,
    categories_evolution,
    threevar_windrose,
    threevar_windrose_trend
)


__all__ = ['quality_control', 
           'compute_climate',
           'compute_daily_records_oneyear',
           'compute_records',
           'plot_records_count',
           'compute_and_plot_exceedances',
           'plot_variable_trends',
           'plot_data_vs_climate',
           'plot_data_vs_climate_withrecords',
           'plot_data_vs_climate_withrecords_multivar',
           'plot_periodstats',
           'plot_data_and_accum_anoms',
           'plot_data_and_annual_cycle',
           'plot_timeseries',
           'timeseries_extremevalues',
           'plot_annual_cycles',
           'get_annual_cycle',
           'annual_meteogram',
           'plot_accumulated_anomalies',
           'plot_anomalies',
           'compare_probdist',
           'categories_evolution',
           'threevar_windrose',
           'threevar_windrose_trend'
           ]
#from .pyclimair import *

#from .pyclimair.clim._clim import compare_with_globaldataset
