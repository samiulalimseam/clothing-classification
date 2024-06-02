import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import requests
from io import BytesIO

# Load the saved model
model = tf.keras.models.load_model("../models/fashion_mnist_model.h5")

# Define class names for Fashion MNIST dataset
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

def preprocess_image(image_path):
    # If the image path is a URL, download the image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content)).convert('L')  # Convert to grayscale
    else:
        img = Image.open(image_path).convert('L')  # Convert to grayscale
        
    img = img.resize((28, 28))  # Resize to match the input shape of the model
    img_array = np.array(img)  # Convert to numpy array
    img_array = img_array / 255.0  # Normalize pixel values
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

def classify_cloth(image_path):
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Get the predicted class label
    predicted_class = np.argmax(predictions[0])
    
    # Print the predicted class label and class name
    print(f"Predicted class label: {predicted_class}")
    print(f"Predicted class name: {class_names[predicted_class]}")
    
    # Plot the image
    if image_path.startswith('http'):
        response = requests.get(image_path)
        img = Image.open(BytesIO(response.content))
    else:
        img = Image.open(image_path)
    plt.imshow(img)
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Prompt the user to enter the image URL
    image_url = input("Enter the image URL: ")
    
    # Call the function to classify the image
    classify_cloth(image_url)
