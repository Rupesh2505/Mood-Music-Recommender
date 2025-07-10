import cv2
import os

# Create folders to store images
os.makedirs("dataset/Happy", exist_ok=True)
os.makedirs("dataset/Sad", exist_ok=True)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
count_happy = 0
count_sad = 0

print("📸 Press 'h' to save Happy image, 's' for Sad, 'q' to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    for (x, y, w, h) in faces:
        # Draw face rectangle
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        face_img = gray[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (48, 48))

        key = cv2.waitKey(1) & 0xFF

        if key == ord('h'):
            cv2.imwrite(f"dataset/Happy/happy_{count_happy}.jpg", face_img)
            count_happy += 1
            print(f"😊 Happy image saved: {count_happy}")
        elif key == ord('s'):
            cv2.imwrite(f"dataset/Sad/sad_{count_sad}.jpg", face_img)
            count_sad += 1
            print(f"😢 Sad image saved: {count_sad}")
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            print("✅ Done collecting images.")
            exit()

    cv2.imshow("Collecting Faces", frame)
