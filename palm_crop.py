import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

image = cv2.imread('imgs/DevHand.jpg')
if image is None:
    print("Error: Image not found.")
    exit()

height, width, _ = image.shape

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

results = hands.process(image_rgb)

if results.multi_hand_landmarks:
    print("Hands detected!")
    for hand_landmarks in results.multi_hand_landmarks:
        palm_landmarks = [0, 1, 2, 5, 9, 13, 17]
        x_min, y_min, x_max, y_max = width, height, 0, 0
        for idx in palm_landmarks:
            landmark = hand_landmarks.landmark[idx]
            x, y = int(landmark.x * width), int(landmark.y * height)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)

        if x_min >= x_max or y_min >= y_max:
            print("Error: Invalid cropping coordinates.")
            continue

        cropped_image = image[y_min:y_max, x_min:x_max]

        if cropped_image.size == 0:
            print("Error: Cropped image is empty.")
        else:
            output_path = 'cropped_palm.jpg'
            if cv2.imwrite(output_path, cropped_image):
                print(f"Image successfully saved as {output_path}")
            else:
                print("Error: Failed to save the image.")
else:
    print("No hands detected.")

hands.close()