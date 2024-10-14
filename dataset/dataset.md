## Brackish Dataset for Underwater Object Detection

### 1. **Dataset Source**
The dataset used for this project is sourced from the **Brackish Dataset** found on Google. It consists of underwater footage, capturing various objects in a brackish environment.

- **Original Data Type**: Video footage of underwater objects.

### 2. **Dataset Preprocessing**

#### 2.1. **Conversion from Video to Frames**
The original dataset was in video format, so the first step was to convert the videos into individual image frames that could be used for training the YOLOv8 model.

**Steps**:
- I used OpenCV's `cv2.VideoCapture()` to extract frames from the videos.
- Frames were saved in a separate directory, one for each video, labeled according to their corresponding video file.


#### 2.2. **Annotation Conversion to YOLO Format**
The next step was to annotate the dataset and convert the annotations into the YOLO-compatible format.

1. **COCO or AAU Conversion**:
   - I used the COCO (Common Objects in Context) format to annotate the objects in each frame.
   - These annotations were then converted into the YOLO format.

2. **YOLO Format**:
   - YOLO requires annotations in `.txt` files for each image, with each line containing:
     - `class_id` (e.g., 0 for fish, 1 for coral, etc.)
     - `x_center` (normalized by image width)
     - `y_center` (normalized by image height)
     - `width` (normalized by image width)
     - `height` (normalized by image height)




### 4. **YOLOv8 Data Configuration File (data.yaml)**

The `data.yaml` file is the configuration file for training YOLOv8. It defines the classes and paths for the images and labels:

```yaml
train: ../train/images
test: ../test/images
val: ../valid/images

nc: 6  
names: ['0', '1', '2', '3', '4', '5']
```

### 5. **Training the YOLOv8 Model**
Once the dataset was prepared, it was used to train the YOLOv8 model. The following command was used for training:

```python
from ultralytics import YOLO

# Load the model
model = YOLO('yolov8n.yaml')  # Load YOLOv8 model

# Train the model
results = model.train(data='path/to/data.yaml', epochs=100)
```
