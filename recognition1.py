import cv2
import mediapipe as mp
import numpy as np
from process.normalizedata import normalize_hand_data1
import joblib
from collections import deque

MODEL_PATH = 'models/hand/best_model.pkl'
LABEL_PATH = 'models/hand/label_encoder.pkl'

model = joblib.load(MODEL_PATH)
label = joblib.load(LABEL_PATH)
id_to_name = {}
for name, ids in label.items():
    for id_ in ids:
        id_to_name[id_] = name

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

prediction_history = deque(maxlen=5)


def predict_gesture(hand_data):
    hand_data = hand_data[3:]
    features = np.array(hand_data).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = np.max(model.predict_proba(features))

    if probability > 0.3:
        prediction_history.append(prediction)

        if len(prediction_history) == prediction_history.maxlen:
            final_prediction = max(set(prediction_history), key=prediction_history.count)
            return final_prediction

    return None


if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                handxyz = []
                for landmark in hand_landmarks.landmark:
                    handxyz.extend([landmark.x, landmark.y, landmark.z])
                handxyz = normalize_hand_data1(handxyz)
                gesture_label = predict_gesture(handxyz)

                if gesture_label is not None:
                    gesture_name = id_to_name.get(gesture_label, "unknown")
                    cv2.putText(image, f"Gesture: {gesture_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gesture Recognition', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
