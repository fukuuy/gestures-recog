import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
from collections import deque
from process.normalizedata import  normalize_hands_data

MODEL_PATH = 'models/hand/dy_model.h5'
LABEL_PATH = 'models/hand/dy_label_encoder.pkl'
model = load_model(MODEL_PATH)
label = joblib.load(LABEL_PATH)
label_map = {v[0]: k for k, v in label.items()}

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)


if __name__ == "__main__":
    SEQUENCE_LEN = 20
    frame_queue = deque(maxlen=SEQUENCE_LEN)
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        h, w, _ = frame.shape
        hands_data = [[], []]
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handxyz = []
                for landmark in hand_landmarks.landmark:
                    handxyz.extend([landmark.x, landmark.y, landmark.z])
                hand_label = 0 if handedness.classification[0].label == 'Left' else 1
                hands_data[hand_label] = handxyz
            hands_data = normalize_hands_data(hands_data)
            frame_queue.append(hands_data)

            if len(frame_queue) == SEQUENCE_LEN:
                sequence = np.array(frame_queue).reshape(1, SEQUENCE_LEN, 60)
                prediction = model.predict(sequence, verbose=0)
                pred_label = np.argmax(prediction)
                gesture = label_map.get(pred_label, "Unknown")
                confidence = prediction[0][pred_label]
                cv2.putText(frame, f"{gesture} ({confidence:.2f})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        cv2.imshow("Dynamic Gesture Recognition", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
