import os
import cv2
import imagehash
import csv
import numpy as np
import pickle
from PIL import Image

# Function to compute perceptual hash
def hash_image(image):
    return imagehash.phash(image)

# Load the CSV file with image hashes
def load_hashes_from_pickle(file_path):
    with open(file_path, mode='rb') as file:
        return pickle.load(file)

# Detect card from the frame
def detect_cards(frame, max_candidates=3):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (5, 5), 0)

    edged = cv2.Canny(blurred, 30, 150)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    card_candidates = []

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    for cnt in contours:
        if len(card_candidates) >= max_candidates:
            break
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * peri, True)
        if 4 <= len(approx) <= 10 and cv2.contourArea(approx) > 5000:
            card_candidates.append(approx)
        
    print(card_candidates)
    return card_candidates


# Load the card image and hash it
def get_card_hash(frame, card_contour):
    pts = card_contour.reshape(4, 2)

    # Sort the points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]      # Top-left
    rect[2] = pts[np.argmax(s)]      # Bottom-right
    rect[1] = pts[np.argmin(diff)]   # Top-right
    rect[3] = pts[np.argmax(diff)]   # Bottom-left

    # Target dimensions (width, height)
    CARD_WIDTH = 250
    CARD_HEIGHT = 350
    dst = np.array([
        [0, 0],
        [CARD_WIDTH - 1, 0],
        [CARD_WIDTH - 1, CARD_HEIGHT - 1],
        [0, CARD_HEIGHT - 1]
    ], dtype="float32")

    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (CARD_WIDTH, CARD_HEIGHT))

    pil_image = Image.fromarray(cv2.cvtColor(warped, cv2.COLOR_BGR2RGB))
    return hash_image(pil_image)

def run():
    # Load hashes from CSV
    image_hashes = load_hashes_from_pickle('image_hashes.pkl')
    # Open webcam
    cap = cv2.VideoCapture(0)
    paused = False
    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect card
            card_contours = detect_cards(frame, max_candidates=3)

            best_match_name = None
            best_match_distance = float('inf')
            best_card_contour = None

            for card_contour in card_contours:
                try:
                    card_hash = get_card_hash(frame, card_contour)
                except:
                    continue  # Skip if transform fails

                for filename, stored_hash in image_hashes.items():
                    distance = card_hash - stored_hash
                    if distance < best_match_distance:
                        best_match_distance = distance
                        best_match_name = filename
                        best_card_contour = card_contour

            MAX_HASH_DISTANCE = float('inf')

            if best_match_name is not None and best_match_distance <= MAX_HASH_DISTANCE:
                cv2.putText(frame, best_match_name, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            if best_card_contour is not None:
                cv2.drawContours(frame, [best_card_contour], -1, (0, 255, 0), 3)

        # Show the video feed with detected card name
        cv2.imshow("Pokémon Card Detector", frame)

        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            paused = False  # Resume when space is pressed

    # Clean up
    cap.release()   
    cv2.destroyAllWindows()
    return

run()