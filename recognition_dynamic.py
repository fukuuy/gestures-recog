import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
import joblib
from collections import deque
from process.normalizedata import normalize_hand_data1, normalize_hands_data


class DynamicGestures:
    def __init__(self, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        self.single_model = load_model('models/hand/dy_model.h5')
        self.single_labels = joblib.load('models/hand/dy_label_encoder.pkl')
        self.single_label_map = {v[0]: k for k, v in self.single_labels.items()}
        self.double_model = load_model('models/hands/dy_model.h5')
        self.double_labels = joblib.load('models/hands/dy_label_encoder.pkl')
        self.double_label_map = {v[0]: k for k, v in self.double_labels.items()}

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.single_queue = deque(maxlen=20)
        self.double_queue = deque(maxlen=20)


    def _get_gesture_name(self, prediction, mode):
        if mode == 'single':
            return self.single_label_map.get(prediction, "unknown")
        else:
            return self.double_label_map.get(prediction, "unknown")

    def predict_single_gesture(self):
        if len(self.single_queue) == self.single_queue.maxlen:
            sequence = np.array(self.single_queue).reshape(1, 20, 60)
            prediction = self.single_model.predict(sequence, verbose=0)
            pred_label = np.argmax(prediction)
            confidence = prediction[0][pred_label]
            return self._get_gesture_name(pred_label, 'single'), confidence
        return None, 0.0

    def predict_double_gesture(self):
        if len(self.double_queue) == self.double_queue.maxlen:
            sequence = np.array(self.double_queue).reshape(1, 20, 126)
            prediction = self.double_model.predict(sequence, verbose=0)
            pred_label = np.argmax(prediction)
            confidence = prediction[0][pred_label]
            return self._get_gesture_name(pred_label, 'double'), confidence
        return None, 0.0

    def process(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        hands_data = [[], []]
        best_gesture = None
        best_confidence = 0.0

        if results.multi_hand_landmarks:
            hand_count = len(results.multi_hand_landmarks)
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                handxyz = []
                for landmark in hand_landmarks.landmark:
                    handxyz.extend([landmark.x, landmark.y, landmark.z])
                hand_label = 0 if handedness.classification[0].label == 'Left' else 1
                hands_data[hand_label] = handxyz
            if hand_count == 1:
                hand_data = hands_data[0] if hands_data[0] else hands_data[1]
                normalized_data = normalize_hand_data1(hand_data)[3:]  # 归一化并去除前3个点
                self.single_queue.append(normalized_data)
                gesture, confidence = self.predict_single_gesture()
                if gesture:
                    best_gesture = gesture
                    best_confidence = confidence
            elif hand_count == 2:
                normalized_data = normalize_hands_data(hands_data)
                left_hand = normalized_data[0] if normalized_data[0] else [0.0] * 63
                right_hand = normalized_data[1] if normalized_data[1] else [0.0] * 63
                self.double_queue.append(left_hand + right_hand)
                gesture, confidence = self.predict_double_gesture()
                if gesture:
                    best_gesture = gesture
                    best_confidence = confidence
        return best_gesture, best_confidence


if __name__ == "__main__":
    dynamic_gestures = DynamicGestures()
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            continue
        gesture, confidence = dynamic_gestures.process(image)
        if gesture:
            cv2.putText(image, f"{gesture} ({confidence:.2f})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
        cv2.imshow('Dynamic Gesture Recognition', image)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC退出
            break

    cap.release()
    cv2.destroyAllWindows()