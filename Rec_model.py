import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
import tkinter as tk
from PIL import Image, ImageTk

# Load the dataset and train the model
df = pd.concat([pd.read_csv(f"D:\project\samples\dataset_{label}.csv") for label in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'])
X = df.drop('label', axis=1)
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=1000)
clf.fit(X_train, y_train)

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create the GUI
root = tk.Tk()
root.title("ASL Recognition")
root.configure(background='#3498db')  # Dark blue background color
root.geometry("800x600")  # Set the initial GUI size

# Create a frame for the webcam feed
webcam_frame = tk.Frame(root, width=640, height=480, bg='#3498db')
webcam_frame.pack(side=tk.TOP, padx=10, pady=10)

# Create a label to display the webcam feed
webcam_label = tk.Label(webcam_frame)
webcam_label.pack()

# Create a frame for the buttons
button_frame = tk.Frame(root, bg='#3498db')
button_frame.pack(side=tk.TOP, padx=10, pady=10)

# Create a button to start capturing
start_button = tk.Button(button_frame, text="Start Capturing", command=lambda: start_capturing(), font=("Helvetica", 16), bg='#2ecc71', fg='white', relief='ridge', borderwidth=2)
start_button.pack(side=tk.LEFT, padx=10, pady=10)

# Create a button to exit
exit_button = tk.Button(button_frame, text="Exit", command=root.destroy, font=("Helvetica", 16), bg='#e74c3c', fg='white', relief='ridge', borderwidth=2)
exit_button.pack(side=tk.LEFT, padx=10, pady=10)

# Create a label to display the detected alphabet
alphabet_label = tk.Label(root, text="", font=("Helvetica", 24), bg='#3498db', fg='white')
alphabet_label.pack(side=tk.BOTTOM, padx=10, pady=10)

# Function to start capturing
def start_capturing():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                landmarks = []
                for landmark in hand_landmarks.landmark:
                    landmarks.extend([landmark.x, landmark.y, landmark.z])
                landmarks = np.array(landmarks).reshape(1, -1)
                prediction = clf.predict(landmarks)
                alphabet_label.config(text=f"Predicted Alphabet: {prediction[0]}")
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img = ImageTk.PhotoImage(img)
        webcam_label.config(image=img)
        webcam_label.image = img
        root.update()

root.mainloop()
