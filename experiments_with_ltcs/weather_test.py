import numpy as np
import pandas as pd
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU

import tensorflow as tf

import ltc_model as ltc
import argparse
import datetime as dt

precip_map = {
    "rain": 0,
    "snow": 1,
    "null": 2
}

def load_trace():
    df = pd.read_csv("data/weather_test/weatherHistory.csv")
    date_time = [dt.datetime.strptime(d, "%Y-%m-%d %H:%M:%S")  for d in date_time]
    temp = df["Temperature (C)"].values.astype(np.float32)
    ap_temp = df["Apparent Temperature (C)"].values.astype(np.float32)
    humidity = df["Humidity"].values.astype(np.float32)
    wind_speed = df["Wind Speed (km/h)"].values.astype(np.float32)
    wind_bearing = df["Wind Bearing (degrees)"].values.astype(np.float32)
    visibility = df["Visibility (km)"].values.astype(np.float32)
    pressure = df["Pressure (millibars)"].values.astype(np.float32)
    precip = df["Precip Type"].values
    precipF = np.empty(precip.shape[0],dtype=np.int32)
    for i in range(precip.shape[0]):
        precipF[i] = precip_map[precip[i]]
    
    