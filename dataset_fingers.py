import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
alphabet_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
df = pd.DataFrame(columns=['label'] + [f'landmark_{i}' for i in range(21*3)])


cap = cv2.VideoCapture(0)

for label in alphabet_labels:
    num_images = int(input(f"Enter the number of images to capture for alphabet {label}: "))
    print(f"Capturing dataset for alphabet {label}...")
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
                df.loc[len(df)] = [label] + landmarks
                cv2.imshow("Capture Dataset", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break 
        if len(df) >= num_images:
            print(f"Capturing complete for alphabet {label}!")
            break
    df.to_csv(f"dataset_{label}.csv", index=False)


cap.release()
cv2.destroyAllWindows()
