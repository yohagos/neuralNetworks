from keras.src.models import Sequential
from keras.src.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from keras.src.optimizers import Adam
from keras.src.utils import image_dataset_from_directory
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau

train_dir = './train'
test_dir = './test'

IMG_SIZE = (200, 200)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 0.00001

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

train_dataset = train_dataset.prefetch(buffer_size=BATCH_SIZE)
test_dataset = test_dataset.prefetch(buffer_size=BATCH_SIZE)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(64, (2, 2)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Conv2D(128, (2, 2)),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
]

history = model.fit(
    train_dataset,
    epochs=EPOCHS,
    validation_data=test_dataset,
    callbacks=callbacks
)

model.save('cat_vs_dog_model.keras')

test_loss, test_acc = model.evaluate(test_dataset)
print(f'Test Accuracy : {test_acc:.5f}')

