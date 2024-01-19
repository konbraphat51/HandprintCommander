import cv2
import mediapipe as mp
from time import time, sleep
import pickle
from HandprintCommander.Utils import draw_keypoints_line
import os

hands = mp.solutions.hands.Hands(
    max_num_hands=2,  
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

#if no data file...
if not os.path.exists("gesture_data.bin"):
    #... create new list
    all_data = []
    
#if data file exists...
else:
    #... load data and append to it
    with open("gesture_data.bin", "rb") as f:
        all_data = pickle.load(f)

#camera
v_cap = cv2.VideoCapture(0)

FPS = 30
FRAME_INTERVAL = 1.0 / FPS


#as long as the camera is open
while v_cap.isOpened():
    start_time = time()
    
    success, img = v_cap.read()
    if not success:
        continue
    
    img = cv2.flip(img, 1)
    
    #get size
    img_h, img_w, _ = img.shape 
    
    # read hand landmarks
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        draw_keypoints_line(
            results,
            img,
        )
        data = {}

        for h_id, hand_landmarks in enumerate(results.multi_hand_landmarks):
            for c_id, hand_class in enumerate(
                results.multi_handedness[h_id].classification
            ):
                positions = [0] * 21
                for idx, lm in enumerate(hand_landmarks.landmark):
                    lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                    positions[idx] = lm_pos
                data[hand_class.label] = positions
      
    # show image
    cv2.imshow("MediaPipe Hands", img)
    
    # key handling
    key = cv2.waitKey(5) & 0xFF
    if key == 27:
        with open("gesture_data.bin", "wb") as f:
            pickle.dump(all_data, f)
        break
    elif key == ord("1"):
        if len(data.keys()) == 2:
            data["Label"] = 1
            all_data.append(data)
            print(len(all_data))
    elif key == ord("0"):
        if len(data.keys()) == 2:
            data["Label"] = 0
            all_data.append(data)
            print(len(all_data))
            
    # control FPS
    elapsed_time = time() - start_time
    sleep_time = max(FRAME_INTERVAL - elapsed_time, 0)
    sleep(sleep_time)

v_cap.release()
