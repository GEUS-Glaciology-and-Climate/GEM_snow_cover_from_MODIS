import pandas as pd
import numpy as np

# Temperature ============================================
# Data update from 2025m - in this there is a gap so I will just use the 2024 data from here to fill on top of the data I was originally working on
zac_air_temp = pd.read_csv('/home/shl/mdrev/data/GEM/zackenberg/Air_temperature_200cm_60_30min_sample_10.17897_G5WS-0W04_data.txt', sep='\t', na_values=-9999)
zac_air_temp['date'] = pd.to_datetime(zac_air_temp['Date'] + ' ' + zac_air_temp['Time'])
zac_air_temp.set_index('date', inplace=True)
zac_air_temp.index = zac_air_temp.index.tz_localize('Etc/GMT')
zac_air_temp.rename(columns={'AT (°C)':'air_temperature_2m'}, inplace = True)
zac_air_temp_2025 = pd.DataFrame(zac_air_temp['air_temperature_2m'].resample('h').last().copy())

# Generate winter precipitation record and PDD
station = 'zackenberg'

#station = 'nuuk'
#station = 'disko'
# --- Load precipitation ---
ds_tp = xr.load_dataset('CARRA/CARRA_tp_nearest_PROMICE.nc')
i = int(np.where(ds_tp['name'].values == station)[0][0])
tp = ds_tp['tp'].isel(station=i).sum(dim="step")  # (time) precip, step=06h+18h

# Convert to mm if dataset uses meters
tp_units = ds_tp['tp'].attrs.get('units', '').lower()
if tp_units in ['m', 'meter', 'metre', 'meters', 'metres']:
    tp = tp * 1000.0
    tp.attrs['units'] = 'mm'

# --- Load temperature ---
ds_t2m = xr.load_dataset('CARRA/CARRA_t2m_nearest_PROMICE.nc')
i = int(np.where(ds_t2m['name'].values == station)[0][0])
t2m = ds_t2m['t2m'].isel(station=i) - 273.15  # °C
t2m_pos = t2m.where(t2m > 0)
PDD = t2m.resample('d').mean()
# --- Align on the time axis (handles any slight mismatches) ---
tp, t2m = xr.align(tp, t2m, join='inner')

# --- Keep precip only when T < 0°C ---
tp_cold = tp.where(t2m < 0)

# --- Calculate number of days with positive temperatures -----
t2m_daymax = t2m.to_pandas().resample('D').max()

positive_days_per_year = (t2m_daymax > 0).groupby(t2m_daymax.index.year).sum().astype(int)
positive_days_per_year.loc[1991:2024].to_csv('results/csvs/CARRA_days_with_positive_temp_'+station+'.csv')
# if you want to treat zeros as positive:
# positive_days_per_year = (t2m_daymax >= 0).groupby(t2m_daymax.index.year).sum().astype(int)

# ---- Hydrological-year totals (September to September)
hyd_year = tp_cold.resample(time='YE-SEP').sum()   # year ends in September
hyd_year = hyd_year.assign_coords(year=hyd_year['time.year'])


# As a pandas (nice for printing/export)
hyd_year_df = pd.DataFrame(hyd_year.to_pandas())
hyd_year_df['year'] = hyd_year_df.index.year

df = hyd_year_df['2000':]
hyd_year_df.to_csv('results/csvs/CARRA_winter_precip_hydro_year_sums_'+station+'.csv')
