import viresclient
import datetime as dt
import pandas as pd
import numpy as np

# According to Lars Toffner-Clausen (2019-02-20):
# Flags_B < 16 "I would always trust"
# Flags_q < 32 "indicates near optimal attitude data"



collections = ['SW_OPER_MAGA_LR_1B', 'SW_OPER_MAGB_LR_1B', 'SW_OPER_MAGC_LR_1B']

request = viresclient.SwarmRequest(url="https://staging.viresdisc.vires.services/openows",
                                   username="karl.laundal",
                                   password="5zA1YNkfO9Qx") 



for collection in collections:
    store = pd.HDFStore(collection + '_raw.h5')
    request.set_collection(collection)
    request.set_products(measurements=["B_NEC", "Flags_B", "Flags_q", "Flags_Platform"],
                         residuals=False,
                         sampling_step="PT30S")

    data = request.get_between(start_time = dt.datetime(2013, 11, 30 ),
                               end_time   = dt.datetime(2019, 1, 1 ))

    xarr = data.as_xarray()

    df = pd.DataFrame({'E_gc': xarr['B_NEC'].values.T[1], 'N_gc':xarr['B_NEC'].values.T[0], 
                       'U_gc':-xarr['B_NEC'].values.T[2], 'gclat':xarr['Latitude'].values,
                       'gclon':xarr['Longitude'].values, 'r_km':xarr['Radius'].values/1000,
                       'Flags_B':xarr['Flags_B'].values, 'Flags_q':xarr['Flags_q'].values},
                       index = xarr['Timestamp'].values)

    for column in df.columns:  
        store.append(column, df[column], format='t')

    store.close()
    print('Saved %s' % (collection + '_raw.h5'))

