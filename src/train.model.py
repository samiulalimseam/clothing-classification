import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist

# Create a directory for saving the dataset
data_dir = "../data/"
os.makedirs(data_dir, exist_ok=True)

# Download and load the Fashion MNIST dataset
train_data, test_data = fashion_mnist.load_data()

# Save the dataset to the data directory
train_images, train_labels = train_data
test_images, test_labels = test_data

# Save training images and labels
with open(os.path.join(data_dir, 'train_images.npy'), 'wb') as f:
    np.save(f, train_images)
with open(os.path.join(data_dir, 'train_labels.npy'), 'wb') as f:
    np.save(f, train_labels)

# Save test images and labels
with open(os.path.join(data_dir, 'test_images.npy'), 'wb') as f:
    np.save(f, test_images)
with open(os.path.join(data_dir, 'test_labels.npy'), 'wb') as f:
    np.save(f, test_labels)

# Preprocess the data
train_images = train_images / 255.0
test_images = test_images / 255.0

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=5, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)

print(f'Test Accuracy: {test_acc}')

# Save the model
model.save("../models/fashion_mnist_model.h5")
