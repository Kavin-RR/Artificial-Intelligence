from numpy import loadtxt
from keras.models import model_from_json

# Load dataset
dataset = loadtxt('diabetes.csv', delimiter=',')
x = dataset[:, 0:8]  # Input
y = dataset[:, 8]    # Output

# Load the model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights('model.weights.h5')
print('Loaded model from disk')

# Make predictions
predictions = model.predict(x)

# Print predictions
for i in range(5, 10):
    print('%s => %d (expected %d)' % (x[i].tolist(), int(predictions[i].item()), int(y[i].item())))
