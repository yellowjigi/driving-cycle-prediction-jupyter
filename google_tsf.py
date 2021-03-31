#!/usr/bin/env python
# coding: utf-8

# ### Tensorflow time series forecasting applied to driving cycle data
# Reference: https://www.tensorflow.org/tutorials/structured_data/time_series

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

# Driving cycle dataset
df = pd.read_csv('data/nedc.csv', names=['t (s)', 'v (km/h)'], index_col='t (s)')

# Have it repeated
df = pd.concat([df, df[1:]], ignore_index=True)
df.index.name = 't (s)'

# Normalize the data
# Use all the data for training (no split)
# df_std = (df - df.mean()) / df.std()
df = (df - df.mean()) / df.std()


# Class for data windowing
class WindowGenerator():
    # 1. Indexes and offsets
    def __init__(self, input_width, label_width, shift,
                 df=df, label_columns=None):
        # Store the raw data.
        self.df = df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(df.columns)}
        
        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift
        
        self.total_window_size = input_width + shift
        
        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]
        
        self.label_start = self.total_window_size - self.label_width
        self.label_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.label_slice]

        # Store example batch of tf.data.Dataset
        self._example = None

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    # 2. Split into inputs/labels
    def split_window(self, features):
        # [batch_size, window_size, features]
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.label_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns], axis=-1)

        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels

    # 3. Plot
    def plot(self, model=None, plot_col='v (km/h)', max_subplots=3):
        inputs, labels = self.example
        plt.figure(figsize=(12, 8))
        plot_col_index = self.column_indices[plot_col] # Here we have only one input column.
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(3, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)

            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()
                  
        plt.xlabel('Time [s]')

    # 4. Create tf.data.Dataset
    # Convert pandas.DataFrame to tensorflow.data.Dataset
    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.df)

    @property
    def example(self):
        result = getattr(self, '_example')
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.train))
            # And cache it for next time
            self._example = result
        return result


MAX_EPOCHS = 32


def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.train,
                        callbacks=[early_stopping])
    
    return history


# Multi-step prediction models

OUT_STEPS = 32
multi_window = WindowGenerator(input_width=32,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

# CNN
# CONV_WIDTH = 3
# multi_conv_model = tf.keras.Sequential([
#     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
#     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
#     # Conv layer
#     # Shape => [batch, 1, conv_units]
#     tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
#     # Output layer
#     # Shape => [batch, 1, out_steps * features]
#     tf.keras.layers.Dense(OUT_STEPS, kernel_initializer=tf.initializers.zeros),
#     # Shape => [batch, out_steps, features]
#     tf.keras.layers.Reshape([OUT_STEPS, 1])])
#
# history = compile_and_fit(multi_conv_model, multi_window)
#
# multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.train)

# RNN
multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS, kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, 1])])

history = compile_and_fit(multi_lstm_model, multi_window)

multi_performance = {'LSTM': multi_lstm_model.evaluate(multi_window.train)}

multi_window.plot(multi_lstm_model)
plt.show()

multi_lstm_model.predict()

# Performance comparison
# x = np.arange(len(multi_performance))
# width = 0.3
#
# metric_index = last_baseline.metrics_names.index('mean_absolute_error')
# mae = [v[metric_index] for v in multi_performance.values()]
#
# plt.bar(x, mae, width)
# plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
# plt.ylabel(f'MAE (average over all times)')
# plt.show()
#
# for name, value in multi_performance.items():
#     print(f'{name:8s}: {value[1]:0.4f}')
