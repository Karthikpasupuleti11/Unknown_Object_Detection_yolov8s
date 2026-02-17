# import supervision as sv

# class Tracker:
#     def __init__(self):
#         self.tracker = sv.ByteTrack(
#             track_thresh=0.4,
#             track_buffer=50,
#             match_thresh=0.9
#         )

#     def update(self, results):
#         detections = sv.Detections.from_ultralytics(results)
#         return self.tracker.update_with_detections(detections)

# tracker.py
import supervision as sv

class Tracker:
    def __init__(self):
        self.tracker = sv.ByteTrack(
            track_thresh=0.25, 
            track_buffer=50,
            match_thresh=0.9
        )

    def update(self, detections):
        # REMOVE: detections = sv.Detections.from_ultralytics(results)
        # The detections are already sv.Detections objects coming from main.py
        return self.tracker.update_with_detections(detections)