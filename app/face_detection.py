import cv2

def detect_faces():
    # Load the pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier('assets/haarcascades/haarcascade_frontalface_default.xml')
    
    # Start capturing video from the webcam
    cap = cv2.VideoCapture(0)
    
    while True:
        # Read the video feed frame by frame
        ret, frame = cap.read()
        
        # Convert the frame to grayscale (Haar Cascade works better on grayscale images)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the image
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Draw a rectangle around each detected face
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        # Display the result
        cv2.imshow("Face Detection", frame)
        
        # Break the loop if 'q' is pressed or the window is closed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Close if 'q' is pressed
            break
        elif cv2.getWindowProperty("Face Detection", cv2.WND_PROP_VISIBLE) < 1:  # Check if the window was closed
            break
    
    # Release the webcam and close the window
    cap.release()
    cv2.destroyAllWindows()

# Call the function to start face detection
if __name__ == "__main__":
    detect_faces()
