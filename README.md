# CNN-Object-Detection-using-SSD-MobileNet-V2

```markdown
# Object Detection using SSD MobileNet V2

## Overview
This project utilizes the pre-trained **SSD MobileNet V2** model for **object detection**. The model is based on **Convolutional Neural Networks (CNN)** and is trained on the COCO dataset to detect various objects in an image. It draws bounding boxes around the detected objects and displays their class labels and detection confidence scores.

## Requirements
Before running the code, ensure you have the following libraries installed:
- TensorFlow
- OpenCV
- NumPy
- Matplotlib

You can install the required libraries using the following command:
```bash
pip install -r requirements.txt
```

## Setup
1. **Download the Pre-trained Model:**
   The model used in this project is SSD MobileNet V2, pre-trained on the COCO dataset. To load the model, make sure to download it from the TensorFlow Hub or use your own model.

   Example:
   ```python
   model = tf.saved_model.load('path_to_your_model_directory')
   ```
   Replace `'path_to_your_model_directory'` with the path to your downloaded model.

2. **Image Input:**
   The script expects an image file to be provided for object detection. The image file is specified in the code as:
   ```python
   image_path = 'car.jpg'  # Specify the path to the car image here
   ```
   - To run the detection on your own image, **replace `'car.jpg'` with the path to your own image file** in the script.

3. **Running the Code:**
   After specifying the image path in the code, you can run the script using:
   ```bash
   python script.py
   ```

4. **Result:**
   - The code will process the image and display the result with bounding boxes around the detected objects along with their class names and detection scores.

## How It Works
- The image is loaded and converted to a format suitable for the model.
- The model detects objects within the image and returns bounding boxes, class IDs, and detection scores.
- The detected objects are then drawn on the image with bounding boxes and class labels.
- The final image is displayed with `matplotlib`.

## Notes
- Ensure that the image file is in the same directory as the script or provide the full path to the image.
- The threshold for object detection can be adjusted. The current threshold is set to **0.5** for detection confidence.
- If you'd like to allow dynamic image uploads (e.g., via a web interface), you will need to integrate a web framework such as Flask or Streamlit.
