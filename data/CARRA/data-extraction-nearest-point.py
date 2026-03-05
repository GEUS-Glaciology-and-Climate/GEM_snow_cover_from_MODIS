# -*- coding: utf-8 -*-
"""
@author: bav@geus.dk

tip list:
    %matplotlib inline
    %matplotlib qt
    import pdb; pdb.set_trace()
"""
import xarray as xr
import cfgrib
import matplotlib.pyplot as plt
import os 
import geopandas as gpd
import pandas as pd
import rioxarray
from rasterio.crs import CRS
import warnings
warnings.filterwarnings("ignore", category=FutureWarning) 

plt.close('all')

# set directory
os.chdir('/mnt/ice/CARRA')
print(os.getcwd())

df = pd.read_csv('/home/shl/tmpdata/CARRA/AWS_station_locations.csv')
# df = pd.read_csv('bav/PROMICE/fbri_test/AWS_KAN_U_M_location.csv')

# this is CARRA projection
crs = CRS.from_string("+units=m +a=6371229.0 +b=6371229.0 +y_0=-9872984.57134 +lon_0=-36.0 +proj=lcc +x_0=-0.0 +lat_2=72.0 +lat_1=72.0")
frac = xr.open_dataset('static/fractions.west.nc')
alt_carra = xr.open_dataset('static/CARRA_orography.grib', engine='cfgrib')
alt_carra['x'] = frac.x
alt_carra['y'] = frac.y
# here extracting at Nuuk
print(df)
df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.lon, df.lat))
df = df.set_crs(4326)
df = df.to_crs(crs)


# variable that should be read
folder_list = ['total_precipitation','2m_temperature']

for folder in folder_list:
    file_list = os.listdir(folder)
    print(folder)

    #to only create a small file as test
    #file_list = [file_list[0], file_list[1]]
    CARRA_nearest_point = []
    for f in file_list:
        if not f.endswith('.grib'):
            continue
        print(f)
        tmp = xr.open_dataset(folder+'/'+f, engine='cfgrib')
        tmp['longitude'] = xr.where(tmp['longitude']>180, 
                                    tmp['longitude']-360,
                                    tmp['longitude'])
        # tmp = tmp.isel(step=1)- tmp.isel(step=0)
        tmp['x'] = frac.x
        tmp['y'] = frac.y
        tmp = tmp.rio.write_crs(crs)
 
        # extracting values at the nearest point
        # print(type(tmp.indexes),tmp.indexes)
        tmp_nearest_point = tmp.interp(x = xr.DataArray(df.geometry.x.to_list(), dims='station'),
                                     y = xr.DataArray(df.geometry.y.to_list(), dims='station'), 
                                    method='nearest')
        
    
        if not CARRA_nearest_point:
            CARRA_nearest_point = tmp_nearest_point
        else:
            CARRA_nearest_point = xr.concat((CARRA_nearest_point, tmp_nearest_point), 
                                            dim='time')
            
    CARRA_nearest_point = CARRA_nearest_point.sortby('time')
    CARRA_nearest_point['latitude'] = ('station', df.lat)
    CARRA_nearest_point['longitude'] = ('station', df.lon)
    CARRA_nearest_point['altitude'] = ('station', df.alt)
    
    CARRA_nearest_point['altitude_mod'] = alt_carra.interp(
        x = xr.DataArray(df.geometry.x.to_list(), dims='station'),
        y = xr.DataArray(df.geometry.x.to_list(), dims='station'), 
        method='nearest').orog
    CARRA_nearest_point['name'] = ('station', df.stid)
    CARRA_nearest_point.to_netcdf('/home/shl/tmpdata/CARRA/CARRA_'+list(tmp.keys())[0]+'_nearest_PROMICE.nc')
    
    # CARRA_nearest_point.to_netcdf('bav/PROMICE/fbri_test/KAN_U_M/CARRA_'+list(tmp.keys())[0]+'_nearest_PROMICE.nc')
