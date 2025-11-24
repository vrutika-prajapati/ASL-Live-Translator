import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math
import time
import os

# Initialize webcam
cap = cv2.VideoCapture(0) # 0 refers to the default webcam
if not cap.isOpened():
    print("Error: Could not open video stream or file. Make sure your webcam is connected and not in use.")
    exit()

detector = HandDetector(maxHands=2, detectionCon=0.8) # Increased detection confidence for better results

offset = 20
imgSize = 300

folder = "Data/ILoveYou"
# Create the folder if it doesn't exist
if not os.path.exists(folder):
    os.makedirs(folder)
    print(f"Created directory: {folder}")

counter = 0

while True:
    success, img = cap.read()
    if not success:
        print("Failed to read from camera. Exiting.")
        break

    # Flip image to prevent mirror effect (optional, but common for webcam feeds)
    img = cv2.flip(img, 1)

    hands, img = detector.findHands(img)

    # Initialize imgWhite as a blank white canvas. It will be updated if hands are detected.
    imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

    processed_hand_present = False # Flag to track if a valid hand image was processed

    if hands: # Check if any hands are detected (one or two)
        # Determine the bounding box based on one or two hands
        if len(hands) == 1:
            x, y, w, h = hands[0]['bbox']
        else: # len(hands) == 2
            hand1, hand2 = hands[0], hands[1]
            x1, y1, w1, h1 = hand1['bbox']
            x2, y2, w2, h2 = hand2['bbox']

            # Calculate combined bounding box for two hands
            x = min(x1, x2)
            y = min(y1, y2)
            w = max(x1 + w1, x2 + w2) - x
            h = max(y1 + h1, y2 + h2) - y

        # Ensure the cropping coordinates are within image boundaries
        y_start = max(0, y - offset)
        y_end = min(img.shape[0], y + h + offset)
        x_start = max(0, x - offset)
        x_end = min(img.shape[1], x + w + offset)

        imgCrop = img[y_start:y_end, x_start:x_end]

        # Only proceed if imgCrop is not empty after boundary checks
        if imgCrop.shape[0] > 0 and imgCrop.shape[1] > 0:
            aspectRatio = h / w

            if aspectRatio > 1: # Height is greater than width (vertical image)
                k = imgSize / h
                wCal = int(w * k) # Corrected: only one k factor
                wCal = min(imgSize, wCal) # Cap wCal to imgSize
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                wGap = (imgSize - wCal) // 2 # Integer division for centering
                imgWhite[:, wGap : wGap + wCal] = imgResize

            else: # Width is greater than or equal to height (horizontal or square image)
                k = imgSize / w
                hCal = int(h * k) # Corrected: only one k factor
                hCal = min(imgSize, hCal) # Cap hCal to imgSize
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                hGap = (imgSize - hCal) // 2 # Integer division for centering
                imgWhite[hGap : hGap + hCal, :] = imgResize
            
            processed_hand_present = True # Set flag to true as we processed a hand image
            cv2.imshow("ImageCrop", imgCrop)
            cv2.imshow("ImageWhite", imgWhite)
        else:
            # If imgCrop is empty after valid bbox, show placeholder
            cv2.imshow("ImageCrop", np.zeros((imgSize, imgSize, 3), np.uint8))
            cv2.imshow("ImageWhite", np.zeros((imgSize, imgSize, 3), np.uint8))
    else:
        # If no hands are detected at all, show placeholder images
        cv2.imshow("ImageCrop", np.zeros((imgSize, imgSize, 3), np.uint8))
        cv2.imshow("ImageWhite", np.zeros((imgSize, imgSize, 3), np.uint8))

    cv2.imshow("Image", img)
    
    key = cv2.waitKey(1) # Use a single waitKey call for responsiveness
    
    if key == ord("s"):
        # Save only if a valid processed hand image was present in imgWhite
        if processed_hand_present:
            counter += 1
            cv2.imwrite(f"{folder}/Image_{time.time()}.jpg", imgWhite)
            print(f"Saved image {counter} to {folder}")
        else:
            print("Cannot save: No valid processed hand image (imgWhite) available, or no hands detected.")

    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()