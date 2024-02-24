import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint
import os

# Загрузка данных
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

# Разделение данных на обучающий и тестовый наборы
train_data, test_data, train_labels, test_labels = train_test_split(data, labels, test_size=0.2, random_state=42)

# Reshape labels
train_labels = train_labels.reshape((-1, 1))
test_labels = test_labels.reshape((-1, 1))

# Создание модели
model = Sequential()
model.add(Dense(128, input_shape=(18, 17), activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation=None))  # No activation in the last layer

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Создание ModelCheckpoint callback
checkpoint_path = "checkpoints/model_checkpoint_epoch_{epoch:02d}.h5"
checkpoint = ModelCheckpoint(checkpoint_path, verbose=1, save_weights_only=True, period=1)

# Обучение модели с использованием callback
model.fit(train_data, train_labels, epochs=10, batch_size=32, validation_split=0.2, callbacks=[checkpoint])

# Оценка модели
accuracy = model.evaluate(test_data, test_labels)[1]
print(f'Test Accuracy: {accuracy}')
