# from ultralytics import YOLO
# import config

# class Detector:
#     def __init__(self):
#         print(f"Loading model: {config.YOLO_MODEL}")
#         self.model = YOLO(config.YOLO_MODEL)
#         self.model.set_classes(config.KNOWN_CLASSES)

#     def detect(self, frame):
#         return self.model.predict(
#             frame,
#             conf=config.CONFIDENCE_THRESHOLD,
#             verbose=False
#         )[0]

# detector.py - FINAL VERSION
from ultralytics import YOLO
import config

class Detector:
    def __init__(self):
        print(f"Loading model: {config.YOLO_MODEL}")
        self.model = YOLO(config.YOLO_MODEL)
        # REMOVED: self.model.set_classes(config.KNOWN_CLASSES)
        # Now detects ALL objects

    def detect(self, frame):
        return self.model.predict(
            frame,
            conf=config.CONFIDENCE_THRESHOLD,
            verbose=False
        )[0]