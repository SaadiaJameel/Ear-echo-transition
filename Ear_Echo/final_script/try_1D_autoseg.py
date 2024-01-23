import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv1D, GlobalMaxPooling1D, Dense

# Generate example data (replace with your labeled data)
X_train = np.random.randn(1000, 100, 1)
y_train = np.random.choice([0, 1], size=(1000, 1), p=[0.9, 0.1])

# Model architecture
input_layer = Input(shape=(100, 1))
conv1d_layer = Conv1D(filters=32, kernel_size=3, activation='relu')(input_layer)
global_pooling_layer = GlobalMaxPooling1D()(conv1d_layer)
dense_layer = Dense(1, activation='sigmoid')(global_pooling_layer)

# Create and compile the model
model = tf.keras.Model(inputs=input_layer, outputs=dense_layer)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)

# Assuming you have a new time series for prediction (replace with your actual data)
new_time_series = np.random.randn(1, 100, 1)

# Use the trained model to predict probabilities
predicted_probabilities = model.predict(new_time_series)

# Threshold the probabilities to identify where the pattern is present
threshold = 0.5
pattern_indices = np.where(predicted_probabilities > threshold)[0]

# Crop the time series around identified pattern segments
cropped_segments = []
for index in pattern_indices:
    start_index = max(0, index - 10)  # Adjust the window size as needed
    end_index = min(100, index + 10)   # Adjust the window size as needed
    cropped_segment = new_time_series[:, start_index:end_index, :]
    cropped_segments.append(cropped_segment)

# The cropped_segments list now contains the identified pattern segments
