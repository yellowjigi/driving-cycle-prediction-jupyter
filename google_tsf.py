#!/usr/bin/env python
# coding: utf-8

# ### Tensorflow time series forecasting applied to driving cycle data
# Reference: https://www.tensorflow.org/tutorials/structured_data/time_series

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

# Driving cycle dataset
df = pd.read_csv('data/nedc.csv', names=['t (s)', 'v (km/h)'], index_col='t (s)')

# Have it repeated
df = pd.concat([df, df[1:]], ignore_index=True)
df.index.name = 't (s)'

# ### Training/Validation/Test = (70%, 20%, 10%)
column_indices = {name: i for i, name in enumerate(df.columns)}

n = len(df)
train_df = df[0:int(n * 0.7)]
val_df = df[int(n * 0.7):int(n * 0.9)]
test_df = df[int(n * 0.9):]

num_features = df.shape[1] # 1

# ### Normalize the data




train_mean = train_df.mean()
train_std = train_df.std()

train_df = (train_df - train_mean) / train_std
val_df = (val_df - train_mean) / train_std
test_df = (test_df - train_mean) / train_std





df_std = (df - train_mean) / train_std
df_std = df_std.melt(var_name='Column', value_name='Normalized')





sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(nrows=2)
fig.tight_layout()
ax = sns.boxplot(x='Normalized', y='Column', data=df_std, ax=axes[0])
ax = sns.violinplot(x='Normalized', y='Column', data=df_std, ax=axes[1])

# plt.show()


# ### Data windowing

# #### 1. Indexes and offsets




class WindowGenerator():
    def __init__(self, input_width, label_width, shift,
                train_df=train_df, val_df=val_df, test_df=test_df,
                label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
        
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
        
    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])


# **Single prediction 24t into the future, given 24t of history**




w1 = WindowGenerator(input_width=24, label_width=1, shift=24, label_columns=['v (km/h)'])


# **Single prediction 1t into the future, given 6t of history**




w2 = WindowGenerator(input_width=6, label_width=1, shift=1, label_columns=['v (km/h)'])


# #### 2. Split into inputs/labels




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

WindowGenerator.split_window = split_window

# Stack three slices by the length of the total window
example_window = tf.stack([np.array(train_df[:w2.total_window_size]),
                          np.array(train_df[10:10 + w2.total_window_size]),
                          np.array(train_df[20:20 + w2.total_window_size])])

example_inputs, example_labels = w2.split_window(example_window)

print('All shapes are: (batch, time, features)')
print(f'Window shape: {example_window.shape}')
print(f'Inputs shape: {example_inputs.shape}')
print(f'Labels shape: {example_labels.shape}')


# 6-timestep & 1-feature inputs to 1-timestep & 1-feature label<br>
# **(Now only the variable is the velocity itself)**

# #### 3. Plot




w2.example = example_inputs, example_labels





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





WindowGenerator.plot = plot
# w2.plot()


# #### 4. Create tf.data.Dataset <br>
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

WindowGenerator.make_dataset = make_dataset

@property
def train(self):
    return self.make_dataset(self.train_df)

@property
def val(self):
    return self.make_dataset(self.val_df)

@property
def test(self):
    return self.make_dataset(self.test_df)

@property
def example(self):
    result = getattr(self, '_example', None)
    if result is None:
        # No example batch was found, so get one from the `.train` dataset
        result = next(iter(self.test))
        # And cache it for next time
        self._example = result
    return result

WindowGenerator.train = train
WindowGenerator.val = val
WindowGenerator.test = test
WindowGenerator.example = example





for example_inputs, example_labels in w2.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


# ### Single step models <br>
# Simplest model




single_step_window = WindowGenerator(
    input_width=1, label_width=1, shift=1,
    label_columns=['v (km/h)'])





for example_inputs, example_labels in single_step_window.train.take(1):
    print(f'Inputs shape (batch, time, features): {example_inputs.shape}')
    print(f'Labels shape (batch, time, features): {example_labels.shape}')


# #### Baseline <br>
# Return the current value as the prediction (i.e., predicting "no change")




class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index
        
    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]
    
baseline = Baseline(label_index=column_indices['v (km/h)'])

baseline.compile(loss=tf.losses.MeanSquaredError(),
                metrics=[tf.metrics.MeanAbsoluteError()])

val_performance = {}
performance = {}

val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
performance['Baseline'] = baseline.evaluate(single_step_window.test)





wide_window = WindowGenerator(
    input_width=32, label_width=32, shift=1,
    label_columns=['v (km/h)'])





# wide_window.plot(baseline)


# #### Linear model <br>
# layers.Dense with no activation set




linear = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1)])

print('Input shape:', single_step_window.example[0].shape)
print('Output shape:', linear(single_step_window.example[0]).shape)





MAX_EPOCHS = 20

def compile_and_fit(model, window, patience=2):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    
    return history

history = compile_and_fit(linear, single_step_window)

val_performance['Linear'] = linear.evaluate(single_step_window.val)
performance['Linear'] = linear.evaluate(single_step_window.test)





# wide_window.plot(linear)


# #### Dense <br>




dense = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1)])

history = compile_and_fit(dense, single_step_window)

val_performance['Dense'] = dense.evaluate(single_step_window.val)
performance['Dense'] = dense.evaluate(single_step_window.test)





# wide_window.plot(dense)


# #### Multi-step dense <br>
# Given multiple time steps as input, predict one time step into the future




CONV_WIDTH = 3
conv_window = WindowGenerator(
    input_width=CONV_WIDTH,
    label_width=1,
    shift=1,
    label_columns=['v (km/h)'])





# conv_window.plot()





multi_step_dense = tf.keras.Sequential([
    # Flatten shape: (time, features) => (time * features)
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1),
    # Add back the time dimension.
    # Shape: (outputs) => (1, outputs)
    tf.keras.layers.Reshape([1, -1])])

print('Input shape:', conv_window.example[0].shape)
print('Output shape:', multi_step_dense(conv_window.example[0]).shape)





history = compile_and_fit(multi_step_dense, conv_window)

val_performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.val)
performance['Multi step dense'] = multi_step_dense.evaluate(conv_window.test)





# conv_window.plot(multi_step_dense)


# #### Convolutional Neural Network




conv_model = tf.keras.Sequential([
    # layers.Flatten & layers.Dense => layers.Conv1D
    tf.keras.layers.Conv1D(filters=32,
                          kernel_size=(CONV_WIDTH),
                          activation='relu'),
    tf.keras.layers.Dense(units=32, activation='relu'),
    tf.keras.layers.Dense(units=1)])

print("Conv model on `conv_window`")
print('Input shape:', conv_window.example[0].shape)
print('Output shape:', conv_model(conv_window.example[0]).shape)





history = compile_and_fit(conv_model, conv_window)

val_performance['Conv'] = conv_model.evaluate(conv_window.val)
performance['Conv'] = conv_model.evaluate(conv_window.test)





print("Wide window")
print('Input shape:', wide_window.example[0].shape)
print('Labels shape:', wide_window.example[1].shape)
print('Output shape:', conv_model(wide_window.example[0]).shape)





LABEL_WIDTH = 32
INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
wide_conv_window = WindowGenerator(
    input_width=INPUT_WIDTH,
    label_width=LABEL_WIDTH,
    shift=1,
    label_columns=['v (km/h)'])






print("Wide conv window")
print('Input shape:', wide_conv_window.example[0].shape)
print('Labels shape:', wide_conv_window.example[1].shape)
print('Output shape:', conv_model(wide_conv_window.example[0]).shape)





# wide_conv_window.plot(conv_model)


# #### Recurrent Neural Network




lstm_model = tf.keras.models.Sequential([
    # Shape: [batch, time, features] => [batch, time, lstm_units]
    tf.keras.layers.LSTM(32, return_sequences=True),
    # Shape: [batch, time, lstm_units] => [batch, time, features]
    tf.keras.layers.Dense(units=1)])





print('Input shape:', wide_window.example[0].shape)
print('Output shape:', lstm_model(wide_window.example[0]).shape)





history = compile_and_fit(lstm_model, wide_window)

val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
performance['LSTM'] = lstm_model.evaluate(wide_window.test)





wide_window.plot(lstm_model)


# #### Performance




x = np.arange(len(performance))

width = 0.3
metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')





val_mae = [v[metric_index] for v in val_performance.values()]
test_mae = [v[metric_index] for v in performance.values()]

plt.ylabel('mean_absolute_error [v (km/h), normalized]')
plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance.keys(), rotation=45)
_ = plt.legend()


# Without the linear model




val_performance_without_linear = {model: v for model, v in val_performance.items()
                                 if model != 'Linear'}
performance_without_linear = {model: v for model, v in performance.items()
                              if model != 'Linear'}





x = np.arange(len(performance_without_linear))

val_mae = [v[metric_index] for v in val_performance_without_linear.values()]
test_mae = [v[metric_index] for v in performance_without_linear.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=performance_without_linear.keys(), rotation=45)
_ = plt.legend()





for name, value in performance.items():
    print(f'{name:12s}: {value[1]:0.4f}')


# ### Multi-output models <br>
# As our data contains only one output (or label), i.e., `v (km/h)`, we will skip the multi-output model

# ### Multi-step models <br>
# Multiple time step predictions




OUT_STEPS = 24
multi_window = WindowGenerator(input_width=24,
                               label_width=OUT_STEPS,
                               shift=OUT_STEPS)

# multi_window.plot()


# #### Baselines <br>
# Repeat the last input time step




class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])

last_baseline = MultiStepLastBaseline()
last_baseline.compile(loss=tf.losses.MeanSquaredError(),
                      metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance = {}
multi_performance = {}

multi_val_performance['Last'] = last_baseline.evaluate(multi_window.val)
multi_performance['Last'] = last_baseline.evaluate(multi_window.test)

# multi_window.plot(last_baseline)


# Repeat the previous time window




class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):
        return inputs

repeat_baseline = RepeatBaseline()
repeat_baseline.compile(loss=tf.losses.MeanSquaredError(),
                        metrics=[tf.metrics.MeanAbsoluteError()])

multi_val_performance['Repeat'] = repeat_baseline.evaluate(multi_window.val)
multi_performance['Repeat'] = repeat_baseline.evaluate(multi_window.test)

# multi_window.plot(repeat_baseline)


# #### Single-shot models <br>
# Predict the entire sequence in a single step <br>
# #### Linear




multi_linear_model = tf.keras.Sequential([
    # Take the last time-step.
    # Shape: [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Output layer
    # Shape => [batch, 1, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])])

history = compile_and_fit(multi_linear_model, multi_window)

multi_val_performance['Linear'] = multi_linear_model.evaluate(multi_window.val)
multi_performance['Linear'] = multi_linear_model.evaluate(multi_window.test)

# multi_window.plot(multi_linear_model)


# #### Dense




multi_dense_model = tf.keras.Sequential([
    # Take the last time step.
    # Shape [batch, time, features] => [batch, 1, features]
    tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    # Hidden layer
    # Shape => [batch, 1, dense_units]
    tf.keras.layers.Dense(512, activation='relu'),
    # Output layer
    # Shape => [batch, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])])

history = compile_and_fit(multi_dense_model, multi_window)

multi_val_performance['Dense'] = multi_dense_model.evaluate(multi_window.val)
multi_performance['Dense'] = multi_dense_model.evaluate(multi_window.test)

# multi_window.plot(multi_dense_model)


# #### CNN




CONV_WIDTH = 3
multi_conv_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    # Conv layer
    # Shape => [batch, 1, conv_units]
    tf.keras.layers.Conv1D(256, activation='relu', kernel_size=(CONV_WIDTH)),
    # Output layer
    # Shape => [batch, 1, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])])

history = compile_and_fit(multi_conv_model, multi_window)

multi_val_performance['Conv'] = multi_conv_model.evaluate(multi_window.val)
multi_performance['Conv'] = multi_conv_model.evaluate(multi_window.test)

# multi_window.plot(multi_conv_model)


# #### RNN




multi_lstm_model = tf.keras.Sequential([
    # Shape [batch, time, features] => [batch, lstm_units]
    # Adding more `lstm_units` just overfits more quickly.
    tf.keras.layers.LSTM(32, return_sequences=False),
    # Shape => [batch, out_steps * features]
    tf.keras.layers.Dense(OUT_STEPS * num_features,
                          kernel_initializer=tf.initializers.zeros),
    # Shape => [batch, out_steps, features]
    tf.keras.layers.Reshape([OUT_STEPS, num_features])])

history = compile_and_fit(multi_lstm_model, multi_window)

multi_val_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.val)
multi_performance['LSTM'] = multi_lstm_model.evaluate(multi_window.test)

multi_window.plot(multi_lstm_model)


# #### Autoregressive model




# class FeedBack(tf.keras.Model):
#     def __init__(self, units, out_steps):
#         super().__init__()
#         self.out_steps = out_steps
#         self.units = units
#         self.lstm_cell = tf.keras.layers.LSTMCell(units)
#         self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
#         self.dense = tf.keras.layers.Dense(num_features)
#     # Init internal state based on the inputs
#     def warmup(self, inputs):
#         # inputs.shape => (batch, time, features)
#         # x.shape => (batch, lstm_units)
#         x, *state = self.lstm_rnn(inputs)
#         # prediction.shape => (batch, features)
#         prediction = self.dense(x)
#         return prediction, state
#
# feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)
# prediction, state = feedback_model.warmup(multi_window.example[0])
# prediction.shape





# def call(self, inputs, training=None):
#     predictions = []
#     # Initialize the LSTM state
#     prediction, state = self.warmup(inputs)
#     # Insert the first prediction
#     predictions.append(prediction)
#
#     # Run the rest of the prediction steps
#     for n in range(1, self.out_steps):
#         # Use the last prediction as input.
#         x = prediction
#         # Execute one LSTM step.
#         x, state = self.lstm_cell(x, states=state, training=training)
#         # Convert the LSTM output to a prediction.
#         prediction = self.dense(x)
#         # Add the prediction to the output
#         predictions.append(prediction)
#
#     # predictions.shape => (time, batch, features)
#     predictions = tf.stack(predictions)
#     # predictions.shape => (batch, time, features)
#     predictions = tf.transpose(predictions, [1, 0, 2])
#     return predictions
#
# FeedBack.call = call





# print('Output shape (batch, time, features): ', feedback_model(multi_window.example[0]).shape)





# history = compile_and_fit(feedback_model, multi_window)
#
# multi_val_performance['AR LSTM'] = feedback_model.evaluate(multi_window.val)
# multi_performance['AR LSTM'] = feedback_model.evaluate(multi_window.test)
#
# multi_window.plot(feedback_model)





x = np.arange(len(multi_performance))
width = 0.3

metric_name = 'mean_absolute_error'
metric_index = lstm_model.metrics_names.index('mean_absolute_error')
val_mae = [v[metric_index] for v in multi_val_performance.values()]
test_mae = [v[metric_index] for v in multi_performance.values()]

plt.bar(x - 0.17, val_mae, width, label='Validation')
plt.bar(x + 0.17, test_mae, width, label='Test')
plt.xticks(ticks=x, labels=multi_performance.keys(), rotation=45)
plt.ylabel(f'MAE (average over all times)')
_ = plt.legend()





for name, value in multi_performance.items():
    print(f'{name:8s}: {value[1]:0.4f}')







