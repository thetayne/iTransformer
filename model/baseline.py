import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from data_provider.data_factory import data_provider
import torch

class BaselineModel:
    def __init__(self, args):
        self.args = args
        self.hourly_averages = None
        self.scaler = None
        #self._get_data()

    def _get_data(self):
        # Load and concatenate all training data batches
        train_data, train_loader = data_provider(self.args, flag='train')
        all_data = []
        for batch_x, _, _, _ in train_loader:
            all_data.append(batch_x.numpy()[:, 0, :])  # Take the first row of each batch
        train_data = np.concatenate(all_data, axis=0)

        print(train_data.shape)
        print(train_data)

        # Scale the data
        self.scaler = StandardScaler()
        train_data_scaled = self.scaler.fit_transform(train_data)

        # Calculate hourly averages
        hourly_averages = []
        for hour in range(24):
            hourly_data = train_data_scaled[hour::24].mean(axis=0)
            hourly_averages.append(hourly_data)
        self.hourly_averages = np.array(hourly_averages)
        print("Data calc.")
        print(self.hourly_averages.shape)
        print(self.hourly_averages)

    def decode_hours(self, encoded_hours):
        # Reverse the encoding: index.hour / 23.0 - 0.5
        decoded_hours = (encoded_hours + 0.5) * 23.0
        return decoded_hours.astype(int)


    def forecast(self, test_loader):
        all_forecasts = []
        for batch_x, batch_y, batch_x_mark, batch_y_mark in test_loader:
            batch_y = batch_y.numpy()
            batch_size, _, num_features = batch_y.shape
            forecast = np.zeros((batch_size, self.args.pred_len, num_features))
            
            # Print shapes for debugging
            # print(f"batch_x shape: {batch_x.shape}")
            # print(f"batch_y shape: {batch_y.shape}")
            # print(f"batch_x_mark shape: {batch_x_mark.shape}")
            # print(f"batch_y_mark shape: {batch_y_mark.shape}")
            
            encoded_test_hours = batch_y_mark[:, -self.args.pred_len:, 0].numpy()
            
            # Print encoded test hours for debugging
            # print(f"encoded_test_hours shape: {encoded_test_hours.shape}")
            # print(f"encoded_test_hours: {encoded_test_hours}")
            
            test_hours = self.decode_hours(encoded_test_hours)  # Decode the hours
            
            # Print decoded test hours for debugging
            # print("Test hours")
            # print(f"test_hours shape: {test_hours.shape}")
            # print(f"test_hours: {test_hours}")
            
            for i in range(batch_size):
                for j in range(self.args.pred_len):
                    hour = test_hours[i, j]
                    forecast[i, j, :] = self.hourly_averages[hour % 24]
            all_forecasts.append(forecast)
        return np.concatenate(all_forecasts, axis=0)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data.reshape(-1, data.shape[-1])).reshape(data.shape)

    def predict(self):
        print("Predicting...")
        _, test_loader = data_provider(self.args, flag='test')
        all_forecasts = []
        all_true = []

        for batch_x, batch_y, _, _ in test_loader:
            batch_y = batch_y.numpy()
            forecast = self.forecast(test_loader)
            forecast = self.inverse_transform(forecast)
            batch_y = self.inverse_transform(batch_y)
            all_forecasts.append(forecast)
            all_true.append(batch_y)

        return np.concatenate(all_forecasts, axis=0), np.concatenate(all_true, axis=0)

    def evaluate(self):
        preds, trues = self.predict()
        mse = ((preds - trues) ** 2).mean()
        mae = np.abs(preds - trues).mean()
        rmse = np.sqrt(mse)
        print(f'MSE: {mse}, MAE: {mae}, RMSE: {rmse}')
        return mse, mae, rmse
