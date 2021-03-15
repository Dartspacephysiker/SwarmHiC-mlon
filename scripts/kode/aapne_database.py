#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 25 10:32:11 2021

@author: qew008
"""

import sys

annadir = '/Data/ift/ift_romfys1/Q1/folk/anna_kvamsdal/kode/'
if not annadir in sys.path:
    sys.path.append(annadir)

import pandas as pd
import numpy as np
from redskap import konveksjonsdatabase
from polarsubplot import Polarsubplot

dbopts = dict(bare_substorm=True)   
sats = ['A','B']
df = []
print("Getting all convection measurements associated with a substorm ...")
for sat in sats:
    df.append(konveksjonsdatabase(sat,**dbopts))
    df[-1]['sat'] = sat
df = pd.concat(df)

