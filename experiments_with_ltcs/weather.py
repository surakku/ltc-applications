import pandas as pd
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # Run on CPU

import tensorflow as tf

import ltc_model as ltc
import argparse
import datetime as dt

def load_trace():
    df = pd.read_csv("data/weather/OHARE_96-23.csv")
    date_time = df["DATE"].values
    alt_setting = df["HourlyAltimeterSetting"].values.astype(np.float32)
    dew_temp = df["HourlyDewPointTemperature"].values.astype(np.float32)
    dry_temp = df["HourlyDryBulbTemperature"].values.astype(np.float32)
    precip = df["HourlyPrecipitation"].values.astype(np.float32)
    pres_change = df["HourlyPressureChange"].values.astype(np.float32)
    humid = df["HourlyRelativeHumidity"].values.astype(np.float32)
    sea_pres = df["HourlySeaLevelPressure"].values.astype(np.float32)
    stat_pres = df["HourlyStationPressure"].values.astype(np.float32)
    df.loc[df['HourlyVisibility'].astype(str).str.contains('V'), 'HourlyVisibility'] = '10.00'
    vis = df["HourlyVisibility"].values.astype(np.float32)
    wet_temp = df["HourlyWetBulbTemperature"].values.astype(np.float32)
    df.loc[df['HourlyWindDirection'].astype(str).str.contains('VRB'), 'HourlyWindDirection'] = '000'
    wind_head = df["HourlyWindDirection"].values.astype(np.float32)
    gust_speed = df["HourlyWindGustSpeed"].values.astype(np.float32)
    wind_speed = df["HourlyWindSpeed"].values.astype(np.float32)
    
    weather_type = df["HourlyPresentWeatherType"].fillna("") ## Consider adding this as a feature somehow, predictions are included
    weather_type = weather_type.str.extract(r"(^\W{0,2}\w\w)").fillna("NONE")
    
    features = np.stack([alt_setting, dew_temp, dry_temp, precip, pres_change, humid, sea_pres, stat_pres, vis, wet_temp, wind_head, gust_speed, wind_speed], axis=-1)
    
    return features, weather_type


def cut_in_sequences(x, y, seq_len, inc=1):
    
    seq_x = []
    seq_y = []
    
    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        seq_x.append(x[start:end])
        seq_y.append(y[start:end])
    return np.stack(seq_x, axis=1), np.stack(seq_y, axis=1)

if __name__ == "__main__":
    load_trace()
