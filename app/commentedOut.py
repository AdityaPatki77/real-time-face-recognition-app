# import cv2
# import face_recognition
# import pickle
# import os
# import threading
# import time

# def load_known_faces():
#     """Load known faces and names from a database (saved pickle file)."""
#     if os.path.exists("known_faces.pkl"):
#         with open("known_faces.pkl", "rb") as f:
#             return pickle.load(f)
#     return {}

# def save_known_faces(known_faces):
#     """Save the updated known faces and names to a pickle file."""
#     with open("known_faces.pkl", "wb") as f:
#         pickle.dump(known_faces, f)

# def recognize_faces(frame, known_faces):
#     """Detect and recognize faces in the given frame."""
#     # Convert the frame to RGB (face_recognition uses RGB images)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Detect faces and face encodings in the image
#     faces = face_recognition.face_locations(rgb_frame, model="hog")
#     encodings = face_recognition.face_encodings(rgb_frame, faces)

#     for (top, right, bottom, left), encoding in zip(faces, encodings):
#         # Check if the detected face matches any known faces
#         matches = face_recognition.compare_faces(list(known_faces.values()), encoding)
#         name = "Unknown"
        
#         # If a match is found, use the corresponding name
#         if True in matches:
#             first_match_index = matches.index(True)
#             name = list(known_faces.keys())[first_match_index]
#         else:
#             # If no match is found, prompt for a new name and add the face to known faces
#             name = input("Enter your name: ")  # This can be replaced by a GUI or more complex input method
#             known_faces[name] = encoding
#             save_known_faces(known_faces)  # Save the updated database

#         # Draw a rectangle around the face and label it
#         cv2.rectangle(frame, (left, top), (right, bottom), (255, 0, 0), 2)
#         cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

# def capture_frames():
#     """Capture video frames in a separate thread for improved performance."""
#     cap = cv2.VideoCapture(0)
#     known_faces = load_known_faces()  # Load known faces from the database
#     frame_skip = 5  # Increase this value to skip more frames
#     prev_time = 0

#     while True:
#         ret, frame = cap.read()

#         if not ret:
#             break

#         # Resize the frame to reduce load (lower resolution)
#         frame_resized = cv2.resize(frame, (640, 480))

#         # Process every nth frame for face recognition
#         if cap.get(cv2.CAP_PROP_POS_FRAMES) % frame_skip == 0:
#             current_time = time.time()
#             # Process face recognition only if enough time has passed (to control the FPS)
#             if current_time - prev_time > 0.1:  # ~10 FPS for recognition
#                 recognize_faces(frame_resized, known_faces)
#                 prev_time = current_time

#         # Display the result
#         cv2.imshow("Face Recognition", frame_resized)

#         # Break the loop if 'q' is pressed or the window is closed
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):  # Close if 'q' is pressed
#             break
#         elif cv2.getWindowProperty("Face Recognition", cv2.WND_PROP_VISIBLE) < 1:  # Check if the window was closed
#             break

#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     # Start the video capture in a separate thread
#     capture_thread = threading.Thread(target=capture_frames)
#     capture_thread.start()
