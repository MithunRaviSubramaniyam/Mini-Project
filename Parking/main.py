import cv2
import pickle
import cvzone
import numpy as np
import time
import os
import easyocr
from datetime import datetime

# Initialize OCR reader
reader = easyocr.Reader(['en'])

# Load resources
cap = cv2.VideoCapture('carPark1.mp4')
img_portrait1 = cv2.imread("carParkPortrait1.png")
img_portrait2 = cv2.imread("carParkPortrait2.png")
output_size = (img_portrait1.shape[1], img_portrait1.shape[0])
img_portrait2 = cv2.resize(img_portrait2, output_size)

# Load position files
with open('CarParkPos_0', 'rb') as f0, open('CarParkPos_1', 'rb') as f1:
    posList_video = pickle.load(f0)
    posList_image = pickle.load(f1)

# Initial mode: video
use_image = False
frame_source = img_portrait2
posList = posList_image

# Reference background from video
_, ref_frame = cap.read()
ref_frame = cv2.resize(ref_frame, output_size)
ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)

# Violation and timing
occupy_start_times = [0] * len(posList_video)
violation_logged = [False] * len(posList_video)
VIOLATION_SECONDS = 15
save_folder = "violations"
os.makedirs(save_folder, exist_ok=True)

def extract_number_plate_text(image):
    result = reader.readtext(image)
    for (bbox, text, prob) in result:
        if len(text) >= 6 and any(c.isdigit() for c in text):
            return text.upper()
    return "UNKNOWN"

def checkParkingSpace(img, ref_gray, posList, mode):
    img_display = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    current_time = time.time()
    free, occupied, violated = 0, 0, 0

    for i, (tl, br) in enumerate(posList):
        x1, y1 = tl
        x2, y2 = br

        crop_current = img[y1:y2, x1:x2]
        crop_ref = ref_gray[y1:y2, x1:x2]
        crop_gray = img_gray[y1:y2, x1:x2]

        if crop_current.shape[0] == 0 or crop_current.shape[1] == 0:
            continue

        motion_score = np.sum(cv2.absdiff(crop_gray, crop_ref))
        color_stddev = np.std(crop_current)
        is_occupied = motion_score > 800000 and color_stddev > 20
        blink = int(current_time * 2) % 2 == 0

        if is_occupied:
            if occupy_start_times[i] == 0:
                occupy_start_times[i] = current_time
            elapsed = current_time - occupy_start_times[i]
            timer_text = f'{int(elapsed // 60):02}:{int(elapsed % 60):02}'

            if elapsed > VIOLATION_SECONDS:
                color = (0, 0, 255) if blink else (0, 0, 150)
                violated += 1

                if not violation_logged[i] and mode == "video":
                    x1_img, y1_img = posList_image[i][0]
                    x2_img, y2_img = posList_image[i][1]
                    snap = img_portrait2[y1_img:y2_img, x1_img:x2_img]
                    timestamp = int(current_time)
                    filename = f"{save_folder}/violation_slot_{i+1}_{timestamp}.png"
                    cv2.imwrite(filename, snap)

                    # OCR
                    plate_text = extract_number_plate_text(snap)
                    log_text = (
                        f"Slot: {i+1}\n"
                        f"Time: {datetime.now()}\n"
                        f"Plate: {plate_text}\n"
                        f"Image: {filename}\n"
                        f"{'-'*30}\n"
                    )
                    with open("violation_records.txt", "a") as f:
                        f.write(log_text)

                    violation_logged[i] = True
            else:
                color = (255, 0, 0)
                occupied += 1
        else:
            occupy_start_times[i] = 0
            violation_logged[i] = False
            color = (0, 255, 0)
            free += 1
            timer_text = ""

        overlay = img_display.copy()
        cv2.rectangle(overlay, tl, br, color, -1)
        cv2.addWeighted(overlay, 0.2, img_display, 0.8, 0, img_display)
        cv2.rectangle(img_display, tl, br, color, 1)

        # Text: slot at top-left, timer at bottom-left
        if is_occupied:
            cv2.putText(img_display, timer_text, (x1 + 5, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(img_display, f"{i+1}", (x1 + 5, y1 + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cvzone.putTextRect(img_display, f'Free: {free}', (10, 40), scale=2, colorR=(0, 255, 0))
    cvzone.putTextRect(img_display, f'Occupied: {occupied}', (10, 90), scale=2, colorR=(255, 0, 0))
    cvzone.putTextRect(img_display, f'Violated: {violated}', (10, 140), scale=2, colorR=(0, 0, 255))

    return img_display

while True:
    if use_image:
        # Show plain image without overlays
        frame_display = img_portrait2.copy()
    else:
        # Read video frame and run full detection
        if cap.get(cv2.CAP_PROP_POS_FRAMES) >= cap.get(cv2.CAP_PROP_FRAME_COUNT) - 1:
            break
        success, frame = cap.read()
        if not success:
            break
        frame = cv2.resize(frame, output_size)
        frame_display = checkParkingSpace(frame, ref_gray, posList_video, "video")

    zoomed = cv2.resize(frame_display, None, fx=1.3, fy=1.3)
    cv2.imshow("Parking Lot Monitor", zoomed)

    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        use_image = not use_image

cap.release()
cv2.destroyAllWindows()