import cv2

def draw_tracks(frame, detections, labels, history):
    if len(detections) == 0:
        return frame

    for box, tid in zip(detections.xyxy, detections.tracker_id):
        x1, y1, x2, y2 = map(int, box)
        cx, cy = (x1 + x2)//2, (y1 + y2)//2
        history[tid].append((cx, cy))

    for box, tid, label in zip(detections.xyxy, detections.tracker_id, labels):
        if label in ["INIT", "CALIBRATING..."]:
            continue

        x1, y1, x2, y2 = map(int, box)
        
        if label == "UNKNOWN":
            box_color, thickness = (0, 0, 255), 3
            text = f"UNKNOWN | ID {tid}"
        else:
            box_color, thickness = (0, 255, 0), 2
            text = f"{label} | ID {tid}"

        # Draw Bounding Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, thickness)

        # Label Background
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), _ = cv2.getTextSize(text, font, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw + 10, y1), box_color, -1)
        cv2.putText(frame, text, (x1 + 5, y1 - 7), font, 0.7, (255, 255, 255), 2)

        # Draw Trajectory for UNKNOWNs
        if label == "UNKNOWN":
            pts = history[tid]
            for i in range(1, len(pts)):
                cv2.line(frame, pts[i-1], pts[i], (0, 0, 255), 2)

    return frame