import os
import tensorflow as tf
from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.src.optimizers import Adam
from keras.src.utils import image_dataset_from_directory
from random import randint

random_num = randint(0, 10000)

train_dir = './train'
test_dir = './test'

IMG_SIZE = (150, 150)
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 0.0001

train_dataset = image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

test_dataset = image_dataset_from_directory(
    test_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='binary'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (2, 2)),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (2, 2)),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset
)

model.save('cat_vs_dog_model'+ str(random_num) +'.keras')

test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test Accuracy : {test_acc:.5f}')

