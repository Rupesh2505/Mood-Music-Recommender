import tkinter as tk
from tkinter import messagebox
import cv2
import pickle
import numpy as np
import webbrowser
import time
import pyttsx3
from PIL import Image, ImageTk

# Load model
with open("model/emotion_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
engine = pyttsx3.init()
engine.setProperty('rate', 150)  # Speed of speech
engine.setProperty('volume', 1)  # Volume level (0.0 to 1.0)


def detect_emotion_and_play_music():
    start_button.config(state="disabled")
    status_label.config(text="🔎 Detecting mood...", fg="#FFD369")
    result_label.config(text="")
    emoji_label.config(image="", text="")

    cap = cv2.VideoCapture(0)
    last_emotion = None
    emotion_start_time = None
    detected = False

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

            current_time = time.time()
            if emotion != last_emotion:
                last_emotion = emotion
                emotion_start_time = current_time
            elif current_time - emotion_start_time >= 2 and not detected:
                cap.release()
                cv2.destroyAllWindows()

                detected = True
                result_label.config(
                    text=f"🧠 Detected Mood: {emotion}",
                    fg="#00fff7" if emotion == "Happy" else "#ff5e78"
                )

                emoji_path = "emoji_happy.png" if emotion == "Happy" else "emoji_sad.png"
                emoji_img = Image.open(emoji_path).resize((120, 120))
                emoji_photo = ImageTk.PhotoImage(emoji_img)
                emoji_label.config(image=emoji_photo)
                emoji_label.image = emoji_photo

                status_label.config(text="🎤 Speaking mood...")

                if emotion == "Happy":
                    engine.say("You look happy today! Let's play something fun.")
                    engine.runAndWait()
                    webbrowser.open("https://www.youtube.com/shorts/eVAkTCeCmYE?feature=share")  # Sunny Dancer shorts
                else:
                    engine.say("You seem a bit sad. Here's something to cheer you up.")
                    engine.runAndWait()
                    webbrowser.open("https://youtu.be/JYodEWUdIso")  # Bewafa nikli tu

                status_label.config(text="✅ Done")
                reset_button.config(state="normal")
                return

        cv2.imshow("🎥 Mood Detection - Press Q to Quit", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    status_label.config(text="❌ Detection stopped.")
    start_button.config(state="normal")


def reset_gui():
    result_label.config(text="")
    status_label.config(text="")
    emoji_label.config(image="", text="")
    start_button.config(state="normal")
    reset_button.config(state="disabled")

    # 🔊 Speak something when reset
    engine.say("Ready to detect your mood again.")
    engine.runAndWait()

    
import tkinter as tk


# --- GUI Setup ---
root = tk.Tk()
root.title("🎧 Mood Music Recommender")
root.geometry("520x500")
root.configure(bg="#0f0c29")  # Deep dark purple background

# Fonts
FONT_TITLE = ("Comic Sans MS", 20, "bold")
FONT_TEXT = ("Comic Sans MS", 13)
FONT_BUTTON = ("Comic Sans MS", 12, "bold")

# Title
title_label = tk.Label(root, text="🎶 Mood Music Recommender", font=FONT_TITLE,
                       bg="#0f0c29", fg="#ffffff")
title_label.pack(pady=25)

# Start Button
start_button = tk.Button(root, text="🎬 Start Detection", font=FONT_BUTTON,
                         bg="#ff416c", fg="white", padx=20, pady=10,
                         activebackground="#ff4b2b", borderwidth=0,
                         command=detect_emotion_and_play_music)
start_button.pack(pady=10)

# Status label
status_label = tk.Label(root, text="", font=FONT_TEXT, bg="#0f0c29", fg="#FFD369")
status_label.pack(pady=5)

# Emoji
emoji_label = tk.Label(root, bg="#0f0c29")
emoji_label.pack(pady=10)

# Result
result_label = tk.Label(root, text="", font=("Comic Sans MS", 16), bg="#0f0c29", fg="#ffffff")
result_label.pack(pady=10)

# Reset Button
reset_button = tk.Button(root, text="🔄 Reset", font=FONT_BUTTON,
                         bg="#393E46", fg="white", padx=15, pady=8,
                         activebackground="#555", borderwidth=0,
                         command=reset_gui, state="disabled")
reset_button.pack(pady=8)

# Footer
footer = tk.Label(root, text="🔧 Powered by OpenCV + Tkinter", font=("Comic Sans MS", 10),
                  bg="#0f0c29", fg="#666")
footer.pack(side=tk.BOTTOM, pady=10)

root.mainloop()
