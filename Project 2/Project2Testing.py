import numpy as np
from keras.preprocessing import image
from keras.models import load_model
import matplotlib.pyplot as plt

# Step 5: Model Testing

# Load the trained model
model = load_model("../model.keras")
classes = ["Crack", "Missing Head", "Paint Off"]

# Function to predict the class and display the image
def display_prediction(img_path):
    img = image.load_img(img_path, target_size=(500, 500))
    img_array = image.img_to_array(img) / 255.0  # Normalize the image data
    img_array = np.expand_dims(img_array, axis=0)

    # Class prediction
    predictions = model.predict(img_array)[0]
    predicted_class = classes[np.argmax(predictions)]
    
    plt.imshow(img)
    plt.axis('off')
    
    # Display prediction text on the image
    text = "\n".join([f"{classes[i]}: {predictions[i]:.2f}" for i in range(len(classes))])
    plt.text(
        10, 450, text, color="green", fontsize=16, 
        bbox=dict(facecolor="white", alpha=0.5)
    )
    plt.title(f"Predicted: {predicted_class}")
    plt.show()

# Testing on new images
display_prediction("/Users/nadir580/Documents/GitHub/data/test/crack/test_crack.jpg")
display_prediction("/Users/nadir580/Documents/GitHub/data/test/missing-head/test_missinghead.jpg")
display_prediction("/Users/nadir580/Documents/GitHub/data/test/paint-off/test_paintoff.jpg")
