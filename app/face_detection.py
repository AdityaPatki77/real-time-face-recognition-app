import cv2
import face_recognition
import pickle
import os
import threading
import time
from queue import Queue
import numpy as np

class FaceRecognitionSystem:
    def __init__(self):
        self.known_faces = self.load_known_faces()
        self.frame_queue = Queue(maxsize=2)
        self.result_queue = Queue(maxsize=2)
        self.processing = True
        self.face_locations_cache = []
        self.last_encodings = []
        self.frame_scale = 0.25
        
    def load_known_faces(self):
        if os.path.exists("known_faces.pkl"):
            with open("known_faces.pkl", "rb") as f:
                return pickle.load(f)
        return {}

    def save_known_faces(self):
        with open("known_faces.pkl", "wb") as f:
            pickle.dump(self.known_faces, f)

    def process_frame(self):
        while self.processing:
            if not self.frame_queue.empty():
                frame = self.frame_queue.get()
                if frame is None:
                    continue

                small_frame = cv2.resize(frame, (0, 0), fx=self.frame_scale, fy=self.frame_scale)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                if len(self.face_locations_cache) == 0:
                    self.face_locations_cache = face_recognition.face_locations(
                        rgb_small_frame, 
                        model="hog",
                        number_of_times_to_upsample=1
                    )
                    self.last_encodings = face_recognition.face_encodings(
                        rgb_small_frame,
                        self.face_locations_cache,
                        num_jitters=1
                    )

                scaled_locations = [
                    (
                        int(top / self.frame_scale),
                        int(right / self.frame_scale),
                        int(bottom / self.frame_scale),
                        int(left / self.frame_scale)
                    )
                    for top, right, bottom, left in self.face_locations_cache
                ]

                results = []
                for encoding in self.last_encodings:
                    matches = []
                    if self.known_faces:
                        matches = face_recognition.compare_faces(
                            list(self.known_faces.values()),
                            encoding,
                            tolerance=0.6
                        )

                    name = "Unknown"
                    if True in matches:
                        first_match_index = matches.index(True)
                        name = list(self.known_faces.keys())[first_match_index]
                    else:
                        name = f"Person_{len(self.known_faces)}"
                        self.known_faces[name] = encoding
                        self.save_known_faces()

                    results.append(name)

                self.result_queue.put((scaled_locations, results))
                if np.random.random() < 0.1:
                    self.face_locations_cache = []
                    self.last_encodings = []

    def run(self):
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)

        process_thread = threading.Thread(target=self.process_frame)
        process_thread.daemon = True
        process_thread.start()

        fps_time = time.time()
        fps_count = 0
        fps = 0

        while self.processing:  # Changed from True to self.processing
            ret, frame = cap.read()
            if not ret:
                break

            fps_count += 1
            if time.time() - fps_time > 1.0:
                fps = fps_count
                fps_count = 0
                fps_time = time.time()

            if not self.frame_queue.full():
                self.frame_queue.put(frame)

            if not self.result_queue.empty():
                locations, names = self.result_queue.get()
                for (top, right, bottom, left), name in zip(locations, names):
                    cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    cv2.rectangle(frame, (left, top - 35), (right, top), (0, 255, 0), cv2.FILLED)
                    cv2.putText(frame, name, (left + 6, top - 6), 
                              cv2.FONT_HERSHEY_DUPLEX, 0.6, (0, 0, 0), 1)

            cv2.putText(frame, f"FPS: {fps}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow('Face Recognition', frame)

            # Fixed window closing logic
            key = cv2.waitKey(1) & 0xFF
            try:
                if key == ord('q') or cv2.getWindowProperty('Face Recognition', cv2.WND_PROP_VISIBLE) < 1:
                    self.processing = False
                    break
            except cv2.error:
                # Window was closed
                self.processing = False
                break

        cap.release()
        cv2.destroyAllWindows()
        # Ensure all windows are closed
        for i in range(1):
            cv2.waitKey(1)

if __name__ == "__main__":
    system = FaceRecognitionSystem()
    system.run()