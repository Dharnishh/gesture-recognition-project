import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.model_selection import train_test_split

# Load the preprocessed data
landmarks_data = np.load('landmarks_data.npy')
labels = np.load('labels.npy')

# Prepare the data
X = np.array(landmarks_data)
y = np.array(labels)

# Reshape X to be 3D (samples, time_steps, features)
X = X.reshape(X.shape[0], 1, X.shape[1])

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = Sequential()
model.add(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(64))
model.add(Dense(32, activation='relu'))
model.add(Dense(len(set(y)), activation='softmax'))  # Number of classes

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Save the model
model.save('gesture_model.h5')  # Ensure this file is created