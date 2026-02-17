# import os

# # Known object categories
# KNOWN_CLASSES = ["person", "car"]

# # Detection threshold (objectness)
# CONFIDENCE_THRESHOLD = 0.25

# # Unknown decision threshold (uncertainty)
# UNKNOWN_CONF_THRESHOLD = 0.4

# # Paths
# INPUT_VIDEO = "/home/cdac/Desktop/unknown_object_tracking/videos/1.mp4"
# OUTPUT_VIDEO = "/home/cdac/Desktop/unknown_object_tracking/outputs/1.mp4"

# # Model
# YOLO_MODEL = "yolov8s-worldv2.pt"

import os

# Known object categories
KNOWN_CLASSES = ["person", "car", "bike"]

# Detection threshold (objectness)
CONFIDENCE_THRESHOLD = 0.25

# Unknown decision threshold (uncertainty)
UNKNOWN_CONF_THRESHOLD = 0.4

# Tracking threshold (for ByteTrack)
TRACKING_CONFIDENCE_THRESHOLD = 0.25

# Paths
INPUT_VIDEO = "/home/cdac/Desktop/unknown_object_tracking/videos/5.mp4"
OUTPUT_VIDEO = "/home/cdac/Desktop/unknown_object_tracking/outputs/5.mp4"

# Model
YOLO_MODEL = "yolov8s-worldv2.pt"

