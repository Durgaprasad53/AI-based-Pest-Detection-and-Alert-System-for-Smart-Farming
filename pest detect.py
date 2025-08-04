from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load the model
model = YOLO("pest.pt")

# Predict on an image
results = model.predict(source="pes1.jpg", save=True)  # Run prediction once

# Get the annotated image
annotated_image = results[0].plot()  # Generate the annotated image

# Function to close the window when 'c' is pressed
def close_on_key(event):
    if event.key == 'c':  # Check if the pressed key is 'c'
        plt.close()  # Close the Matplotlib window

# Display the image using Matplotlib
fig, ax = plt.subplots()
ax.imshow(annotated_image)  # Display the annotated image
ax.axis('off')  # Turn off axis for a cleaner display
fig.canvas.mpl_connect('key_press_event', close_on_key)  # Connect the key press event
plt.show()  # Keeps the image displayed until a key is pressed