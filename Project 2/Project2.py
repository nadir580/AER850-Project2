# Part 1: Data Processing
import tensorflow as tf
from keras.preprocessing import image_dataset_from_directory
from keras import layers, models

IMAGE_SIZE = (500, 500)
CHANNELS = 3
INPUT_SHAPE = (*IMAGE_SIZE, CHANNELS)

train_data_dir = "/Users/nadir580/Documents/GitHub/Data/train"
validation_data_dir = "/Users/nadir580/Documents/GitHub/Data/valid"
test_data_dir = "/Users/nadir580/Documents/GitHub/Data/test"

train_dataset = image_dataset_from_directory(
    train_data_dir,
    image_size=IMAGE_SIZE,
    batch_size=32,
    label_mode='categorical'
)

validation_dataset = image_dataset_from_directory(
    validation_data_dir,
    image_size=IMAGE_SIZE,
    batch_size=32,
    label_mode='categorical'
)

test_dataset = image_dataset_from_directory(
    test_data_dir,
    image_size=IMAGE_SIZE,
    batch_size=32,
    label_mode='categorical'
)

augment = models.Sequential([
    layers.Rescaling(1./255),
    layers.RandomZoom(0.1),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(factor=0.1)
])

train_dataset = train_dataset.map(lambda x, y: (augment(x, training=True), y))
validation_dataset = validation_dataset.map(lambda x, y: (layers.Rescaling(1./255)(x), y))

autotune = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=autotune)
validation_dataset = validation_dataset.prefetch(buffer_size=autotune)
test_dataset = test_dataset.prefetch(buffer_size=autotune)

for x_batch, y_batch in train_dataset.take(1):
    print("Batch Shape: ", x_batch.shape, y_batch.shape)

# Part 2: Neural Network Architecture Design
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation='relu', input_shape=INPUT_SHAPE))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Part 3: Hyperparameter Analysis
from tensorflow.keras.layers import LeakyReLU, ELU

model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), strides=(1, 1), activation=None, input_shape=INPUT_SHAPE))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), strides=(1, 1), activation=None))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), strides=(1, 1), activation=None))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())

model.add(Dense(128, activation=None))
model.add(ELU(alpha=1.0))
model.add(Dropout(0.5))

model.add(Dense(3, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


# Part 4: Model Evaluation
fitted_model = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
axes[0].plot(fitted_model.history['accuracy'], label='Training Accuracy')
axes[0].plot(fitted_model.history['val_accuracy'], label='Validation Accuracy')
axes[0].set_title('Model Accuracy')
axes[0].legend()

axes[1].plot(fitted_model.history['loss'], label='Training Loss')
axes[1].plot(fitted_model.history['val_loss'], label='Validation Loss')
axes[1].set_title('Model Loss')
axes[1].legend()

plt.show()

model.save("../model.keras")
test_loss, test_accuracy = model.evaluate(test_dataset)
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")


