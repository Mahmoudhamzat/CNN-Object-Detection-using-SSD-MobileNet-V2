# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 19:37:21 2025

@author: lenovo
"""
#https://www.kaggle.com/models/tensorflow/ssd-mobilenet-v2/tensorFlow2/ssd-mobilenet-v2/1?tfhub-redirect=true



import os
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras

# List of category names supported by the SSD MobileNet v2 model trained on COCO
category_names = {1: 'Person', 2: 'Bicycle', 3: 'Car', 4: 'Motorcycle', 5: 'Airplane',
                  6: 'Bus', 7: 'Train', 8: 'Truck', 9: 'Boat', 10: 'Traffic light',
                  11: 'Fire hydrant', 13: 'Dog', 14: 'Horse', 15: 'Sheep', 16: 'Cow',
                  17: 'Elephant', 18: 'Bear', 19: 'Zebra', 20: 'Giraffe', 21: 'Backpack',
                  22: 'Umbrella', 23: 'Handbag', 24: 'Tie', 25: 'Suitcase', 26: 'Frisbee',
                  27: 'Skis', 28: 'Snowboard', 29: 'Sports ball', 30: 'Kite', 31: 'Baseball bat',
                  32: 'Baseball glove', 33: 'Skateboard', 34: 'Surfboard', 35: 'Tennis racket',
                  36: 'Bottle', 37: 'Wine glass', 38: 'Cup', 39: 'Fork', 40: 'Knife',
                  41: 'Spoon', 42: 'Bowl', 43: 'Banana', 44: 'Apple', 45: 'Sandwich', 46: 'Orange',
                  47: 'Broccoli', 48: 'Carrot', 49: 'Hot dog', 50: 'Pizza', 51: 'Donut', 52: 'Cake',
                  53: 'Chair', 54: 'Couch', 55: 'Potted plant', 56: 'Bed', 57: 'Dining table',
                  58: 'Toilet', 59: 'Tv', 60: 'Laptop', 61: 'Mouse', 62: 'Remote', 63: 'Keyboard',
                  64: 'Cell phone', 65: 'Microwave', 66: 'Oven', 67: 'Toaster', 68: 'Sink',
                  69: 'Refrigerator', 70: 'Book', 71: 'Clock', 72: 'Vase', 73: 'Scissors',
                  74: 'Teddy bear', 75: 'Hair drier', 76: 'Toothbrush'}

# Load pre-trained SSD model from TensorFlow Hub
model = tf.saved_model.load('G:\yolo')  # Make sure to load the correct model

# Load the image
image_path = 'car.jpg'  # Specify the path to the car image here
image = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Convert image to Tensor
input_tensor = tf.convert_to_tensor(image_rgb)
input_tensor = input_tensor[tf.newaxis,...]#(H,W,3)

# Perform object detection
detections = model(input_tensor)

# Extract detection results
boxes = detections['detection_boxes'][0].numpy()
class_ids = detections['detection_classes'][0].numpy().astype(int)
scores = detections['detection_scores'][0].numpy()

# Define the threshold for detecting objects
threshold = 0.5  # Confidence threshold for object detection
for i in range(len(scores)):
    if scores[i] > threshold:
        box = boxes[i]
        y_min, x_min, y_max, x_max = box

        # Convert the coordinates to the image format
        (startX, startY, endX, endY) = (int(x_min * image.shape[1]), int(y_min * image.shape[0]), 
                                         int(x_max * image.shape[1]), int(y_max * image.shape[0]))

        # Extract the class name using the class ID
        class_id = class_ids[i]
        class_name = category_names.get(class_id, "Unknown")  # Use the name from the dictionary

        # Draw a rectangle around the object
        cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
        
        # Add the label (class name and score)
        cv2.putText(image, f"{class_name}: {scores[i]:.2f}", (startX, startY - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

# Display the image with detections
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
plt.imshow(image_bgr)
plt.show()
