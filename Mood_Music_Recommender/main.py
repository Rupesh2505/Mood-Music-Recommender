import cv2
import pickle
import numpy as np
import webbrowser
import time

# Load trained model
with open("model/emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

# Tracking variables
last_emotion = None
emotion_start_time = None
music_played = False

print("🎥 Webcam running... Show an emotion. Press 'q' to quit early.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        face_img = gray[y:y+h, x:x+w]
        face_resized = cv2.resize(face_img, (48, 48))
        face_flattened = face_resized.flatten().reshape(1, -1)

        prediction = model.predict(face_flattened)[0]
        emotion = "Happy" if prediction == 1 else "Sad"

        # Draw face and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        current_time = time.time()

        # Start or reset emotion timer
        if emotion != last_emotion:
            last_emotion = emotion
            emotion_start_time = current_time
            music_played = False

        # Play music if emotion stays stable for 3 seconds
        elif not music_played and current_time - emotion_start_time >= 3:
            music_played = True

            if emotion == "Happy":
                print("😊 Happy detected! Opening music...")
                webbrowser.open("https://www.youtube.com/watch?v=ZbZSe6N_BXs")  # Happy - Pharrell Williams
            else:
                print("😢 Sad detected! Opening music...")
                webbrowser.open("https://www.youtube.com/watch?v=YQHsXMglC9A")  # Sad - Adele

            # 🛑 Exit after playing one song
            cap.release()
            cv2.destroyAllWindows()
            print("🛑 Done. Music opened, exiting.")
            exit()

    cv2.imshow("Mood Music Recommender", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()





    