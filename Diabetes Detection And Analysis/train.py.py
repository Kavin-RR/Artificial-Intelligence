'''
A - Pregnancies
B - Glucose
C - BloodPressure
D - SkinThickness
E - Insulin
F - BMI
G - DiabetesPedigreeFunction
H - Age
I - Outcome

'''

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense,Input
from keras.models import model_from_json

dataset = loadtxt('diabetes.csv',delimiter = ',')
#print(dataset)

x = dataset[:,0:8]#input
y = dataset[:,8]#output

#print('input',x)
#print('output',y)

model = Sequential()

model.add(Input(shape=(8,)))
model.add(Dense(12, activation='relu'))  # First hidden layer
model.add(Dense(8, activation='relu'))   # Second hidden layer
model.add(Dense(1, activation='sigmoid'))  # Output layer

model.compile(loss = 'binary_crossentropy',optimizer = 'adam',metrics = ['accuracy'])

model.fit(x,y,epochs =70,batch_size=10)#to train the model

#evaluation
_,accuracy = model.evaluate(x,y)
print('Accuracy: %2f' % (accuracy*100))

#model save the files
model_json =model.to_json()
with open("model.json","w") as json_file:
    json_file.write(model_json)
model.save_weights("model.weights.h5")
print("saved model to disk")



