"""
RIPPED FROM DAG repository
SMH 2021/02/18

Script to Download, read, resample and create a data frame to use the Penticton solar radio flux data.
The Ottawa radio flux data set (historical f10.7 data set) stops in 1991. After that period of time data from 
the Penticton solar flux dataset are used. The Penticton data set contains three values of solar radio flux 
for three different times. 1700, 2000 (local noon) and 2300UT/ summer. 1800, 2000 and 2200UT /winter.
The Ottawa data set used only one data point per day, values at around noon. In this script the
Penticton values are resampled for one day using the median of the three values. This is in accordance with 
what done by NOAA between 1991 and 2018. They used the nearest to noon value, unless a radio burst is 
detected at that time, in which case another value on that day is used. 

For more info on the historical data set see:
http://lasp.colorado.edu/lisird/data/noaa_radio_flux/
For more info on the Penticton data set see:
https://lasp.colorado.edu/lisird/data/penticton_radio_flux/

"""
import requests
import pandas as pd
import numpy as np

'''
Donwload data in csv format for Ottawa and Penticton Adjusted radio flux and save them in a folder
The path is arbitrary, you can also use the command input() (now commented) to choose a path. 
The Penticton data set comes also with the observed radio flux. 
'''

filepath = '/SPENCEdata/Research/database/SHEIC/'

##Penticton
time_stamp = '2021-02-05T08:41:00.000Z'#arbitrary time, it automatically chooses the last available
file_url_penticton = 'https://lasp.colorado.edu/lisird/latis/dap/penticton_radio_flux.csv?&time>=1947-01-01T00:00:00.000Z&time<=' + time_stamp
#filepath=input('Where would you like to save the data: \n')
filename_penticton= 'penticton_radio_flux.csv'
filedata = requests.get(file_url_penticton)
open(filepath + filename_penticton, 'wb').write(filedata.content)
print('Penticton file downloaded...')

#print(1/0)
'''
Read the data
'''

#read Penticton
f107new = pd.read_csv(filepath + filename_penticton, sep = ',', parse_dates= True, index_col = 0)
f107new[f107new == 0] = np.nan


######### Convert Julian to datetime 
time = np.array(f107new.index)
epoch = pd.to_datetime(0, unit = 's').to_julian_date()
time = pd.to_datetime(time-epoch, unit = 'D')
#########

###set datetime as index
f107new = f107new.reset_index()
f107new.set_index(time, inplace=True)

'''
Resample the Penticton data set for one day- 
the Ottawa data set has one measurement per day, the Penticton has 3 measurements per day
'''
f107new_min = f107new.resample('1D').min()
f107new_mean = f107new.resample('1D').mean()
f107new_median = f107new.resample('1D').median()

###Drop the Nan's for the f107new_median, we are interested in the median. 
###The same can be done for min and mean
f107new_median = f107new_median.dropna()

##Drop Julian date column 
f107new_median = f107new_median.drop(columns=['time (Julian Date)'], axis = 1)

#print(1/0)
'''
Extract the column of interest
'''
f107observed_flux = f107new_median[f107new_median.keys()[0]].values
f107adjusted_flux = f107new_median[f107new_median.keys()[1]].values

'''
Other option: create a pandas data frame with only the column/colums of interest
'''
f107Penticton = pd.DataFrame(index = f107new_median.index)
f107Penticton['adjusted solar radio flux'] =  f107new_median[f107new_median.keys()[1]].values
