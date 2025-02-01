import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.models import model_from_json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Load dataset
dataset = np.loadtxt('diabetes.csv', delimiter=',')
X = dataset[:, 0:8]  # Input features
Y = dataset[:, 8]    # Output (Target)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Standardize the input features to normalize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the model
model = Sequential()
model.add(Input(shape=(8,)))                # Input layer
model.add(Dense(16, activation='relu'))     # First hidden layer
model.add(Dense(12, activation='relu'))     # Second hidden layer
model.add(Dense(8, activation='relu'))      # Third hidden layer
model.add(Dense(1, activation='sigmoid'))   # Output layer

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, Y_train, epochs=100, batch_size=10, verbose=1)

# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, Y_test)
print(f'Accuracy on test data: {accuracy * 100:.2f}%')

# Save the model
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")
print("Saved model to disk")
