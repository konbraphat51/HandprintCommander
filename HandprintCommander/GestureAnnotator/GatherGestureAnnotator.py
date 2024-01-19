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
        
    print("imported existing data")

#camera
v_cap = cv2.VideoCapture(0)

FPS = 30
FRAME_INTERVAL = 1.0 / FPS


def _read_hand():
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # if hands detected
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
                 
        return data

    # if no hands detected
    else:
        return None

label_current = 0

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
    data_hand = _read_hand()
    if data_hand == None:
        flag_detected = False
    else:
        flag_detected = True
      
    # show image
    cv2.imshow("MediaPipe Hands", img)
    
    # key handling
    key = cv2.waitKey(5) & 0xFF
    
    # if "q" key is pressed...
    if key == 113:
        #... save and exit
        with open("gesture_data.bin", "wb") as f:
            pickle.dump(all_data, f)
            
        print("Saved")
        
        break
    
    # if space key is pressed...
    elif key == 32:
        #... register data
        data_hand["Label"] = label_current
        all_data.append(data_hand)
        print(f"Registered: {len(all_data)}")
        
    # if "r" key is pressed...
    elif key == 114:
        #... cancel the previous registration
        if len(all_data) > 0:
            all_data.pop()
        print("Canceled")
        
    # if "i" key is pressed...
    elif key == 105:
        #... change label
        try:
            label_current = int(input("new label number (integer) >> "))
            print(f"Label changed to {label_current}")
        except:
            print("invalid integer!")
            
    # control FPS
    elapsed_time = time() - start_time
    sleep_time = max(FRAME_INTERVAL - elapsed_time, 0)
    sleep(sleep_time)

v_cap.release()
