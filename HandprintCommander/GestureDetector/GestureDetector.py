from time import time, sleep
import tensorflow as tf
import numpy as np
import cv2
import mediapipe as mp
from HandprintCommander.Utils import preprocessing_gesture_data, draw_keypoints_line


FPS = 30
FRAME_INTERVAL = 1.0 / FPS
INFERING_THRESHOLD = 0.9

# initialize TFLite model
interpreter = tf.lite.Interpreter(model_path="model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

hands = mp.solutions.hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.4, 
    min_tracking_confidence=0.4, 
)


def _infer_gesture(data):
    processed_data = preprocessing_gesture_data([data]).astype(np.float32)
    interpreter.set_tensor(input_details[0]["index"], processed_data)
    interpreter.invoke()
    
    # result is save in output_details
    output_data = interpreter.get_tensor(output_details[0]["index"])

    # find the class larger than threshold
    output_data = np.array(output_data[0])
    output_data = np.where(output_data > INFERING_THRESHOLD)
    return output_data

def _read_hands(img, img_w, img_h):
    results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.multi_hand_landmarks:
        draw_keypoints_line(
            results,
            img,
        )

        # データの追加に関する処理
        data = {}
        for h_id, hand_landmarks in enumerate(
            results.multi_hand_landmarks
        ):
            for c_id, hand_class in enumerate(
                results.multi_handedness[h_id].classification
            ):
                positions = [0] * 21
                for idx, lm in enumerate(hand_landmarks.landmark):
                    lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                    positions[idx] = lm_pos
                data[hand_class.label] = positions
                
        if (("Right" in data) and ("Left" in data)):
            return data
        else:
            return None
        
    else:
        return None

def capture(on_detected:callable):
    #camera
    v_cap = cv2.VideoCapture(0) 

    # as long as the video is not finished
    while v_cap.isOpened():
        start_time = time()
        
        success, img = v_cap.read()
        if not success:
            continue
        
        img = cv2.flip(img, 1)
        
        # get size
        img_h, img_w, _ = img.shape 
        
        # hand detection
        data_hand = _read_hands(img, img_w, img_h)
        if data_hand is not None:
            output_label = _infer_gesture(data_hand)
            
            # if gesture detected...
            if output_label != 0:
                on_detected(output_label)

        cv2.imshow("MediaPipe Hands", img)

        # control FPS
        elapsed_time = time() - start_time
        sleep_time = max(FRAME_INTERVAL - elapsed_time, 0)
        time_delta = max(elapsed_time, FRAME_INTERVAL)
        sleep(sleep_time)

if __name__ == "__main__":
    capture(print)