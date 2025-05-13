import cv2
import mediapipe as mp
import numpy as np
from process.normalizedata import normalize_hands_data
import joblib
from collections import deque

MODEL_PATH = 'models/hands/best_model.pkl'
LABEL_PATH = 'models/hands/label_encoder.pkl'

model = joblib.load(MODEL_PATH)
label = joblib.load(LABEL_PATH)
id_to_name = {}
for name, ids in label.items():
    for id_ in ids:
        id_to_name[id_] = name

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

prediction_history = deque(maxlen=5)


def predict_gesture(hand_data):
    features = np.array(hand_data).reshape(1, -1)
    prediction = model.predict(features)[0]
    probability = np.max(model.predict_proba(features))

    if probability > 0.4:
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

        hands_data = [[], []]
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handxyz = []
                for landmark in hand_landmarks.landmark:
                    handxyz.extend([landmark.x, landmark.y, landmark.z])
                hand_label = 0 if handedness.classification[0].label == 'Left' else 1
                hands_data[hand_label] = handxyz
            hands_data = normalize_hands_data(hands_data)

            if hands_data[0] and hands_data[1]:
                gesture_label = predict_gesture(hands_data)

                if gesture_label is not None:
                    gesture_name = id_to_name.get(gesture_label, "unknown")
                    cv2.putText(image, f"Gesture: {gesture_name}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow('Gesture Recognition', image)  # cv2.flip(image, 1)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
