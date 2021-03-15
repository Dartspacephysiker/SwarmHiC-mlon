import pandas as pd

infile = '/home/spencerh/Desktop/substorms-ohtani-20131201_000000_to_20211202_000000.ascii'

from datetime import datetime
interper = lambda y,m,d,H,M: datetime.strptime(y+m+d+H+M,"%Y%m%d%H%M")
#names=['mlt','mlat','glon','glat']
names=['yr','mo','day','h','m','mlt','mlat','glon','glat']
ss = pd.read_csv(infile,sep='\s+',skiprows=38,header=None,infer_datetime_format=True,parse_dates={'time': [0,1,2,3,4]},date_parser=interper,names=names)

