# import cv2
# import os
# from collections import defaultdict, deque

# import config
# from detector import Detector
# from tracker import Tracker
# from unknown_filter import is_uncertain
# from utils import draw_tracks

# # =========================
# # TRACKING & STABILITY
# # =========================
# UNKNOWN_SCORE_THRESHOLD = 5
# RECOVERY_SCORE = 2
# MIN_TRACK_AGE = 5
# CLASS_LOCK_THRESHOLD = 6   # frames needed to lock class

# def main():
#     detector = Detector()
#     tracker = Tracker()

#     if not os.path.exists(config.INPUT_VIDEO):
#         print("ERROR: Input video not found")
#         return

#     cap = cv2.VideoCapture(config.INPUT_VIDEO)
#     w, h = int(cap.get(3)), int(cap.get(4))
#     fps = cap.get(cv2.CAP_PROP_FPS)

#     out = cv2.VideoWriter(
#         config.OUTPUT_VIDEO,
#         cv2.VideoWriter_fourcc(*'mp4v'),
#         fps,
#         (w, h)
#     )

#     # =========================
#     # TRACK STATE
#     # =========================
#     track_history = defaultdict(lambda: deque(maxlen=50))
#     unknown_score = defaultdict(int)
#     track_age = defaultdict(int)

#     # 🔒 CLASS LOCKING
#     track_class_votes = defaultdict(lambda: defaultdict(int))
#     track_final_class = {}

#     frame_id = 0

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_id += 1

#         # =========================
#         # 1. DETECT
#         # =========================
#         results = detector.detect(frame)

#         if results.boxes is None or len(results.boxes) == 0:
#             out.write(frame)
#             continue

#         # 🔴 HARD FILTER (CRITICAL)
#         keep = results.boxes.conf.cpu().numpy() >= 0.45
#         results.boxes = results.boxes[keep]

#         if len(results.boxes) == 0:
#             out.write(frame)
#             continue

#         # =========================
#         # 2. TRACK
#         # =========================
#         tracked = tracker.update(results)

#         if len(tracked) == 0:
#             out.write(frame)
#             continue

#         # =========================
#         # 3. CLASS LOCK + UNKNOWN
#         # =========================
#         final_labels = []

#         for conf, cls, tid in zip(
#             tracked.confidence,
#             tracked.class_id,
#             tracked.tracker_id
#         ):
#             track_age[tid] += 1
#             class_name = config.KNOWN_CLASSES[int(cls)]

#             # 🚫 Ignore unstable tracks
#             if track_age[tid] < MIN_TRACK_AGE:
#                 final_labels.append("INIT")
#                 continue

#             # 🔒 CLASS VOTING
#             track_class_votes[tid][class_name] += 1

#             # Lock class if enough votes
#             if tid not in track_final_class:
#                 best_class = max(
#                     track_class_votes[tid],
#                     key=track_class_votes[tid].get
#                 )
#                 if track_class_votes[tid][best_class] >= CLASS_LOCK_THRESHOLD:
#                     track_final_class[tid] = best_class

#             # Use locked class if exists
#             label = track_final_class.get(tid, class_name)

#             # UNKNOWN logic (track-level)
#             if is_uncertain(conf):
#                 unknown_score[tid] += 1
#             else:
#                 unknown_score[tid] = max(
#                     0, unknown_score[tid] - RECOVERY_SCORE
#                 )

#             if unknown_score[tid] >= UNKNOWN_SCORE_THRESHOLD:
#                 final_labels.append("UNKNOWN")
#             else:
#                 final_labels.append(label)

#         # =========================
#         # 4. DRAW
#         # =========================
#         annotated = draw_tracks(
#             frame,
#             tracked,
#             final_labels,
#             track_history
#         )

#         out.write(annotated)
#         print(f"Processing frame {frame_id}", end="\r")

#     cap.release()
#     out.release()
#     print(f"\nDone. Output saved to {config.OUTPUT_VIDEO}")

# if __name__ == "__main__":
#     main()
import cv2
import os
import numpy as np
import supervision as sv
from collections import defaultdict, deque

import config
from detector import Detector
from tracker import Tracker
from utils import draw_tracks

# =========================
# BALANCED STABILITY SETTINGS
# =========================
UNKNOWN_SCORE_THRESHOLD = 8   
RECOVERY_SCORE = 2            
MIN_TRACK_AGE = 8             # Number of frames before showing label
CLASS_LOCK_THRESHOLD = 5      
CONSISTENCY_RATIO = 0.5       

class SequentialIDManager:
    def __init__(self):
        self.byte_to_seq = {}
        self.next_seq_id = 1
        self.active_tracks = set()
        
    def update(self, byte_track_ids):
        current_ids = set(byte_track_ids)
        new_ids = current_ids - self.active_tracks
        for new_byte_id in new_ids:
            if new_byte_id not in self.byte_to_seq:
                self.byte_to_seq[new_byte_id] = self.next_seq_id
                self.next_seq_id += 1
        self.active_tracks = current_ids
        return [self.byte_to_seq[bid] for bid in byte_track_ids]

def main():
    detector = Detector()
    tracker = Tracker()
    id_manager = SequentialIDManager()

    cap = cv2.VideoCapture(config.INPUT_VIDEO)
    if not cap.isOpened(): return
    
    w, h, fps = int(cap.get(3)), int(cap.get(4)), cap.get(cv2.CAP_PROP_FPS)
    os.makedirs(os.path.dirname(config.OUTPUT_VIDEO), exist_ok=True)
    out = cv2.VideoWriter(config.OUTPUT_VIDEO, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))

    # Track States
    track_history = defaultdict(lambda: deque(maxlen=30))
    unknown_score = defaultdict(int)
    track_age = defaultdict(int)
    track_class_votes = defaultdict(lambda: defaultdict(int))
    track_locked_class = {}
    track_conf_history = defaultdict(lambda: deque(maxlen=15))

    frame_id = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_id += 1

        results = detector.detect(frame)
        if results.boxes is None or len(results.boxes) == 0:
            out.write(frame)
            continue

        detections = sv.Detections.from_ultralytics(results)
        tracked = tracker.update(detections)
        
        if len(tracked) == 0:
            out.write(frame)
            continue

        sequential_ids = id_manager.update(tracked.tracker_id.tolist())
        final_labels = []

        for i, tid in enumerate(sequential_ids):
            track_age[tid] += 1
            curr_conf = tracked.confidence[i]
            curr_name = results.names[tracked.class_id[i]]
            
            track_conf_history[tid].append(curr_conf)
            avg_conf = sum(track_conf_history[tid]) / len(track_conf_history[tid])

            if track_age[tid] < MIN_TRACK_AGE:
                final_labels.append("INIT")
                continue

            # Class Voting
            if curr_name in config.KNOWN_CLASSES:
                track_class_votes[tid][curr_name] += 1

            # Locking logic
            if tid not in track_locked_class:
                best_class = max(track_class_votes[tid], key=track_class_votes[tid].get, default=None)
                if best_class and track_class_votes[tid][best_class] >= CLASS_LOCK_THRESHOLD:
                    track_locked_class[tid] = best_class

            display_name = track_locked_class.get(tid, "CALIBRATING...")

            # Unknown Logic
            if curr_name not in config.KNOWN_CLASSES or avg_conf < config.UNKNOWN_CONF_THRESHOLD:
                unknown_score[tid] += 2
            else:
                unknown_score[tid] = max(0, unknown_score[tid] - RECOVERY_SCORE)

            if unknown_score[tid] >= UNKNOWN_SCORE_THRESHOLD:
                final_labels.append("UNKNOWN")
            else:
                final_labels.append(display_name.upper())

        # Counts for Dashboard
        k_count = sum(1 for l in final_labels if l not in ["INIT", "UNKNOWN", "CALIBRATING..."])
        u_count = sum(1 for l in final_labels if l == "UNKNOWN")

        # Drawing
        tracked_seq = sv.Detections(xyxy=tracked.xyxy, confidence=tracked.confidence, 
                                    class_id=tracked.class_id, tracker_id=np.array(sequential_ids))
        annotated = draw_tracks(frame, tracked_seq, final_labels, track_history)

        # TOP-LEFT DASHBOARD
        overlay = annotated.copy()
        cv2.rectangle(overlay, (10, 10), (300, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)
        
        cv2.putText(annotated, f"LIVE TRACKING", (20, 40), 1, 1.5, (255, 255, 0), 2)
        cv2.putText(annotated, f"KNOWN:   {k_count}", (20, 75), 1, 1.2, (0, 255, 0), 2)
        cv2.putText(annotated, f"UNKNOWN: {u_count}", (20, 105), 1, 1.2, (0, 0, 255), 2)

        out.write(annotated)
        print(f"Frame {frame_id} processed...", end="\r")

    cap.release()
    out.release()
    print("\nProcessing complete. Video saved.")

if __name__ == "__main__":
    main()