import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Dense
import os

dirname = "/home/bektemir/Desktop/my_projects/faceRecognation/dataset"
data_files = os.listdir(dirname)
data = []
labels = []

for file in data_files:
    with open(dirname + '/' + file, 'r') as f:
        data_array = np.array([list(map(float, line.split())) for line in f.readlines()])
        data.append(data_array)
        labels.append(1 if file.split('-')[0] == file.split('-')[1] else 0)

data = np.array(data)
labels = np.array(labels)

train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

train_labels = train_labels.reshape((-1, 1))
test_labels = test_labels.reshape((-1, 1))


model = Sequential()
model.add(Dense(128, input_shape=(18, 17), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_path = '/home/bektemir/Desktop/my_projects/faceRecognation/checkpoints/model_checkpoint_epoch_10.h5'

model.load_weights(checkpoint_path)

accuracy = model.evaluate(test_data, test_labels)[1]
print(f'Accuracy for {checkpoint_path}: {accuracy}')
