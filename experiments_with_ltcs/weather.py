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
    alt_setting = normalize(df["HourlyAltimeterSetting"]).fillna(0).values.astype(np.float32) ##TODO Normalzie the inputs brrrother https://datascience.stackexchange.com/questions/39916/loss-function-returns-nan-on-time-series-dataset-using-tensorflow
    dew_temp = normalize(df["HourlyDewPointTemperature"]).fillna(0).values.astype(np.float32)
    dry_temp = normalize(df["HourlyDryBulbTemperature"]).fillna(0).values.astype(np.float32)
    precip = normalize(df["HourlyPrecipitation"]).fillna(0).values.astype(np.float32)
    pres_change = normalize(df["HourlyPressureChange"]).fillna(0).values.astype(np.float32)
    humid = normalize(df["HourlyRelativeHumidity"]).fillna(0).values.astype(np.float32)
    sea_pres = normalize(df["HourlySeaLevelPressure"]).fillna(0).values.astype(np.float32)
    stat_pres = normalize(df["HourlyStationPressure"]).fillna(0).values.astype(np.float32)
    df.loc[df['HourlyVisibility'].astype(str).str.contains('V'), 'HourlyVisibility'] = '10.00'
    vis = normalize(df["HourlyVisibility"].astype(np.float32)).fillna(0).values
    wet_temp = normalize(df["HourlyWetBulbTemperature"]).fillna(0).values.astype(np.float32)
    df.loc[df['HourlyWindDirection'].astype(str).str.contains('VRB'), 'HourlyWindDirection'] = '000'
    wind_head = normalize(df["HourlyWindDirection"].astype(np.float32)).fillna(0).values
    gust_speed = normalize(df["HourlyWindGustSpeed"]).fillna(0).values.astype(np.float32)
    wind_speed = normalize(df["HourlyWindSpeed"]).fillna(0).values.astype(np.float32)
    
    df["HourlyPresentWeatherType"] = df["HourlyPresentWeatherType"].str.extract(r"(^\W{0,2}\w\w)").fillna("NONE")
    weather_col = pd.Categorical(df["HourlyPresentWeatherType"], categories=df["HourlyPresentWeatherType"].fillna("NONE").unique()).codes ## Consider adding this as a feature somehow, predictions are included
    df["HourlyPresentWeatherType"] = weather_col
    weather_type = df["HourlyPresentWeatherType"].values.astype(np.float32)
    
    features = np.stack([alt_setting, dew_temp, dry_temp, precip, pres_change, humid, sea_pres, stat_pres, vis, wet_temp, wind_head, gust_speed, wind_speed], axis=-1)
    
    return features, weather_type

## Min Max normalization
def normalize(data):
    min_val = np.min(data)
    max_val = np.max(data)

    scaled_data = (data - min_val) / (max_val - min_val)

    return(scaled_data)


def cut_in_sequences(x, y, seq_len, inc=1):
    
    seq_x = []
    seq_y = []
    
    for s in range(0, x.shape[0] - seq_len, inc):
        start = s
        end = start + seq_len
        seq_x.append(x[start:end])
        seq_y.append(y[start:end])
    return np.stack(seq_x, axis=1), np.stack(seq_y, axis=1)


class WeatherData():
    def __init__(self, seq_len=64):
        
        x, y = load_trace()
        
        train_x, train_y = cut_in_sequences(x, y, seq_len, inc=4) ## Consider increasing increment
        
        self.train_x = np.stack(train_x, axis=0)
        self.train_y = np.stack(train_y, axis=0)
        total_seqs = self.train_x.shape[1]
        print("Total number of training sequences: {}".format(total_seqs))
        permutation = np.random.RandomState(176126).permutation(total_seqs)
        valid_size = int(0.1 * total_seqs)
        test_size = int(0.15 * total_seqs)
        
        self.valid_x = self.train_x[:, permutation[:valid_size]]
        self.valid_y = self.train_y[:, permutation[:valid_size]]
        self.test_x = self.train_x[:, permutation[valid_size : valid_size + test_size]]
        self.test_y = self.train_y[:, permutation[valid_size : valid_size + test_size]]
        self.train_x = self.train_x[:, permutation[valid_size + test_size :]]
        self.train_y = self.train_y[:, permutation[valid_size + test_size :]]
        
        
    def iterate_train(self, batch_size=16):
        total_seqs = self.train_x.shape[1]
        permutation = np.random.permutation(total_seqs)
        total_batches = total_seqs // batch_size
            
        for i in range(total_batches):
            start = i * batch_size
            end = start + batch_size
            batch_x = self.train_x[:, permutation[start:end]]
            batch_y = self.train_y[:, permutation[start:end]]

            yield (batch_x, batch_y)
                
class WeatherModel:
    def __init__(self, model_type, model_size, learning_rate=0.01):
        self.model_type = model_type
        self.constrain_op = None
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, None, 13])
        self.target_y = tf.placeholder(dtype=tf.float32, shape=[None, None])
        
        self.model_size = model_size
        head = self.x
        
        self.wm = ltc.LTCCell(model_size)
        if model_type.endswith("_rk"):
            self.wm._solver = ltc.ODESolver.RungeKutta
        elif model_type.endswith("_ex"):
            self.wm._solver = ltc.ODESolver.Explicit
        else:
            self.wm._solver = ltc.ODESolver.SemiImplicit
            
        head, _ = tf.nn.dynamic_rnn(
            self.wm, head, dtype=tf.float32, time_major=True
        )
        self.constrain_op = self.wm.get_param_constrain_op()
        
        target_y = tf.expand_dims(self.target_y, axis=-1)

        self.y = tf.layers.Dense(
            1,
            activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(),
        )(head)
        
        print("logit shape: ", str(self.y.shape))
        self.loss = tf.reduce_mean(tf.square(target_y - self.y))
        optimizer = tf.train.AdamOptimizer(learning_rate)
        self.train_step = optimizer.minimize(self.loss)

        self.accuracy = tf.reduce_mean(tf.abs(target_y - self.y))

        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())
        
        self.result_file = os.path.join(
            "results", "weather", "{}_{}.csv".format(model_type, model_size)
        )
        if not os.path.exists("results/weather"):
            os.makedirs("results/weather")
        if not os.path.isfile(self.result_file):
            with open(self.result_file, "w") as f:
                f.write(
                    "best epoch, train loss, train mae, valid loss, valid mae, test loss, test mae\n"
                )

        self.checkpoint_path = os.path.join(
            "tf_sessions", "weather", "{}".format(model_type)
        )
        if not os.path.exists("tf_sessions/weather"):
            os.makedirs("tf_sessions/weather")

        self.saver = tf.train.Saver()
        
    def save(self):
        self.saver.save(self.sess, self.checkpoint_path)

    def restore(self):
        self.saver.restore(self.sess, self.checkpoint_path)
        
    def fit(self, data, epochs, verbose=True, log_period=10):

        best_valid_loss = np.PINF
        best_valid_stats = (0, 0, 0, 0, 0, 0, 0)
        self.save()
        
        for e in range(epochs):
            if verbose and e % log_period == 0:
                test_acc, test_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: data.test_x, self.target_y: data.test_y},
                )
                valid_acc, valid_loss = self.sess.run(
                    [self.accuracy, self.loss],
                    {self.x: data.valid_x, self.target_y: data.valid_y},
                )
                # MSE metric -> less is better
                if (valid_loss < best_valid_loss and e > 0) or e == 1:
                    best_valid_loss = valid_loss
                    best_valid_stats = (
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                    self.save()
            
            losses = []
            accs = []
            for batch_x, batch_y in data.iterate_train(batch_size=16):
                acc, loss, _ = self.sess.run(
                    [self.accuracy, self.loss, self.train_step],
                    {self.x: batch_x, self.target_y: batch_y},
                )
                if not self.constrain_op is None:
                    self.sess.run(self.constrain_op)

                losses.append(loss)
                accs.append(acc)

            if verbose and e % log_period == 0:
                print(
                    "Epochs {:03d}, train loss: {:0.2f}, train mae: {:0.2f}, valid loss: {:0.2f}, valid mae: {:0.2f}, test loss: {:0.2f}, test mae: {:0.2f}".format(
                        e,
                        np.mean(losses),
                        np.mean(accs),
                        valid_loss,
                        valid_acc,
                        test_loss,
                        test_acc,
                    )
                )
            if e > 0 and (not np.isfinite(np.mean(losses))):
                break
        self.restore()
        (
            best_epoch,
            train_loss,
            train_acc,
            valid_loss,
            valid_acc,
            test_loss,
            test_acc,
        ) = best_valid_stats
        print(
            "Best epoch {:03d}, train loss: {:0.3f}, train mae: {:0.3f}, valid loss: {:0.3f}, valid mae: {:0.3f}, test loss: {:0.3f}, test mae: {:0.3f}".format(
                best_epoch,
                train_loss,
                train_acc,
                valid_loss,
                valid_acc,
                test_loss,
                test_acc,
            )
        )
        with open(self.result_file, "a") as f:
            f.write(
                "{:08d}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}, {:0.8f}\n".format(
                    best_epoch,
                    train_loss,
                    train_acc,
                    valid_loss,
                    valid_acc,
                    test_loss,
                    test_acc,
                )
            )
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ltc")
    parser.add_argument("--log", default=1, type=int)
    parser.add_argument("--size", default=64, type=int)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()

    weather_data = WeatherData()
    model = WeatherModel(model_type=args.model, model_size=args.size)

    model.fit(weather_data, epochs=args.epochs, log_period=args.log)
