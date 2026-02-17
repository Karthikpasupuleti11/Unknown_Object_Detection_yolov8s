from ultralytics import YOLO
import cv2
import os

# =========================
# CONFIGURATION
# =========================
MODEL_PATH = "yolov8s-worldv2.pt"
# Hardcode your single image path here
IMAGE_PATH = "/home/cdac/Desktop/unknown_object_tracking/image copy 2.png"

def main():
    # 1. Load the model 
    # (By NOT using set_classes, it uses its default 80+ class vocabulary)
    model = YOLO(MODEL_PATH)

    if not os.path.exists(IMAGE_PATH):
        print(f"Error: Image not found at {IMAGE_PATH}")
        return

    # 2. Run inference
    # The model will look for everything it knows (Horse, Zebra, Person, etc.)
    results = model.predict(IMAGE_PATH, conf=0.25, verbose=False)[0]

    # 3. Print the findings
    print(f"\n--- Model Findings for {os.path.basename(IMAGE_PATH)} ---")
    if len(results.boxes) == 0:
        print("No objects detected.")
    else:
        for box in results.boxes:
            class_id = int(box.cls[0])
            label = results.names[class_id]
            confidence = float(box.conf[0])
            print(f"Detected: {label.upper()} | Confidence: {confidence:.2f}")

    # 4. Display the visual result
    annotated_image = results.plot()
    cv2.imshow("Automatic Detection", annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()