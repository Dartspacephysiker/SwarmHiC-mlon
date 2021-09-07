lats = np.r_[-60:60]
lons = np.r_[0:360:1]
LAT,LON = np.meshgrid(lats,lons)
import apexpy
from datetime import datetime
a = apexpy.Apex(datetime(2020,1,1),110)
swarmalt = 500
mlat,mlon = a.geo2apex(LAT,LON,swarmalt)
print(f"Min   mlat for glat range [-60:60:1], glon range [0:360:10]: {np.min(np.abs(mlat))}")
print(f"N NaN mlat for glat range [-60:60:1], glon range [0:360:10]: {np.sum(np.isnan(mlat))}")
