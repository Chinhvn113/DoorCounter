# opencv_tracker_engine.py

import cv2
import uuid

# For newer versions of OpenCV, tracker objects are created differently.
# This function handles the creation for both old and new versions.
def create_tracker():
    """
    Creates a CSRT tracker instance, handling API changes across different
    OpenCV versions in a more robust way.
    """
    # Check if the 'legacy' submodule exists. This is true for older versions.
    if hasattr(cv2, 'legacy'):
        # We are on an older version of OpenCV
        return cv2.legacy.TrackerKCF_create()
    else:
        # We are on a newer version of OpenCV (4.5.4+)
        return cv2.TrackerKCF_create()

class OpenCVTrackerEngine:
    """
    Manages multiple OpenCV trackers for detected objects.
    """
    def __init__(self, max_lost_frames=10):
        self.trackers = {}  # Stores {id: (tracker, box)}
        self.max_lost_frames = max_lost_frames
        self.lost_frames_count = {} # Stores {id: count}

    def update_trackers(self, frame):
        """
        Updates the position of all active trackers on a new frame.
        """
        updated_boxes = {}
        lost_ids = []

        for track_id, (tracker, _) in self.trackers.items():
            success, box = tracker.update(frame)
            if success:
                # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                x, y, w, h = [int(v) for v in box]
                x1, y1, x2, y2 = x, y, x + w, y + h
                updated_boxes[track_id] = (x1, y1, x2, y2)
                self.lost_frames_count[track_id] = 0 # Reset lost count on success
            else:
                self.lost_frames_count[track_id] += 1
                if self.lost_frames_count[track_id] > self.max_lost_frames:
                    lost_ids.append(track_id)

        # Remove lost trackers
        for track_id in lost_ids:
            self.remove_tracker(track_id)

        return updated_boxes

    def add_tracker(self, frame, box):
        """
        Initializes and adds a new tracker for a given bounding box.
        """
        tracker = create_tracker()
        # Convert from (x1, y1, x2, y2) to (x, y, w, h) for initialization
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        tracker.init(frame, (x1, y1, w, h))

        track_id = str(uuid.uuid4())[:8] # Generate a unique ID
        self.trackers[track_id] = (tracker, box)
        self.lost_frames_count[track_id] = 0
        return track_id

    def remove_tracker(self, track_id):
        """Removes a tracker by its ID."""
        if track_id in self.trackers:
            del self.trackers[track_id]
            del self.lost_frames_count[track_id]

    def get_tracked_objects(self):
        """Returns the current bounding boxes of all tracked objects."""
        return {tid: box for tid, (_, box) in self.trackers.items()}