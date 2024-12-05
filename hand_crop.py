import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

image = cv2.imread('imgs/DevHand.jpg')
height, width, _ = image.shape

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    for hand_landmarks in results.multi_hand_landmarks:
        x_min, y_min, x_max, y_max = width, height, 0, 0
        for landmark in hand_landmarks.landmark:
            x, y = int(landmark.x * width), int(landmark.y * height)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        cropped_image = image[y_min:y_max, x_min:x_max]

        cv2.imwrite('cropped_hand.jpg', cropped_image)

hands.close()
