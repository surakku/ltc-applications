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
    "null": 2,
    "": 3
}

summary_map = {
    "Partly Cloudy": 0,
    "Mostly Cloudy": 1,
    "Overcast": 2,
    "Breezy and Mostly Cloudy": 3,
    "Clear": 4,
    "Breezy and Partly Cloudy": 5,
    "Breezy and Overcast": 6,
    "Humid and Mostly Cloudy": 7,
    "Humid and Partly Cloudy": 8,
    "Windy and Foggy": 9,
    "Windy and Overcast": 10,
    "Breezy and Foggy": 11,
    "Windy and Partly Cloudy": 12,
    "Breezy": 13,
    "Dry and Partly Cloudy": 14,
    "Windy and Mostly Cloudy": 15,
    "Dangerously Windy and Partly Cloudy": 16,
    "Dry": 17,
    "Windy": 18,
    "Humid and Overcast": 19,
    "Light Rain": 20,
    "Drizzle": 21,
    "Windy and Dry": 22,
    "Dry and Mostly Cloudy": 23,
    "Breezy and Dry": 24,
    "Rain": 25,
    "Foggy": 26
}

def load_trace():
    df = pd.read_csv("data/weather_test/weatherHistory.csv")
    date_time = df["Formatted Date"].values
    date_time = [dt.datetime.strptime(d[:-10], "%Y-%m-%d %H:%M:%S")  for d in date_time]
    temp = df["Temperature (C)"].values.astype(np.float32)
    ap_temp = df["Apparent Temperature (C)"].values.astype(np.float32)
    humidity = df["Humidity"].values.astype(np.float32)
    wind_speed = df["Wind Speed (km/h)"].values.astype(np.float32)
    wind_bearing = df["Wind Bearing (degrees)"].values.astype(np.float32)
    visibility = df["Visibility (km)"].values.astype(np.float32)
    pressure = df["Pressure (millibars)"].values.astype(np.float32)
    precip = df["Precip Type"].fillna("").values
    precipF = np.empty(precip.shape[0],dtype=np.int32)
    for i in range(precip.shape[0]):
        precipF[i] = precip_map[precip[i]]
    summary = df["Summary"].values
    summaryF = np.empty(summary.shape[0], dtype=np.int32)
    for i in range(summary.shape[0]):
        summaryF[i] = summary_map[summary[i]]
    
    features = np.stack([date_time, temp, ap_temp, humidity, wind_speed, wind_bearing, visibility, pressure, precipF, summaryF], axis=-1)
    
        
    return features[:, :-1], features[:, -1]
    
    
if(__name__ == "__main__"):
    features, summ = load_trace()
    
    