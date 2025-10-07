import cv2
import easyocr
import os
import re

# Haarcascade file
harcascade = "model/haarcascade_russian_plate_number.xml"

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # width
cap.set(4, 480)  # height

min_area = 500
count = 1

# EasyOCR Reader
reader = easyocr.Reader(['en'])

# TN District Code Mapping (add more as needed)
tn_districts = {
    '01': 'Chennai Central',
    '02': 'Chennai North',
    '03': 'Chennai South',
    '04': 'Chengalpattu',
    '05': 'Kanchipuram',
    '06': 'Tiruvallur',
    '07': 'Vellore',
    '09': 'Salem',
    '10': 'Namakkal',
    '11': 'Dharmapuri',
    '12': 'Erode',
    '13': 'Coimbatore',
    '14': 'Tiruppur',
    '15': 'Nilgiris',
    '18': 'Trichy',
    '19': 'Thanjavur',
    '20': 'Tiruvarur',
    '21': 'Nagapattinam',
    '22': 'Madurai',
    '23': 'Dindigul',
    '24': 'Sivagangai',
    '25': 'Ramanathapuram',
    '28': 'Tirunelveli',
    '29': 'Thoothukudi',
    '30': 'Kanyakumari',
    '72': 'Tiruvallur (Ambattur RTO)',
    # Add more codes if needed
}

# Regex pattern to match TN plates like TN 72 AB 1234
plate_pattern = re.compile(r'TN\s?(\d{2})\s?[A-Z]{1,2}\s?\d{4}')

while True:
    success, img = cap.read()
    plate_cascade = cv2.CascadeClassifier(harcascade)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    for (x, y, w, h) in plates:
        area = w * h
        if area > min_area:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
            img_roi = img[y: y + h, x: x + w]
            cv2.imshow("ROI", img_roi)

    cv2.imshow("Result", img)

    key = cv2.waitKey(1)

    if key == ord('s'):
        # Save the ROI image
        img_path = f"/Users/mithunravi/Desktop/Number-Plate/plates/scaned_img_{count}.jpg"
        cv2.imwrite(img_path, img_roi)
        
        # Feedback
        cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, "Plate Saved", (150, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 2, (0, 0, 255), 2)
        cv2.imshow("Result", img)
        cv2.waitKey(500)

        # === OCR ===
        output = reader.readtext(img_path)
        detected_plate = None

        for _, text, conf in output:
            if conf > 0.3:
                text = text.replace("IND", "").replace(" ", "")
                match = plate_pattern.search(text)
                if match:
                    code = match.group(1)
                    district = tn_districts.get(code, "Unknown District")
                    formatted_plate = f"TN{code} {text[-8:-4]} {text[-4:]}"  # format TN72 AB 1234
                    detected_plate = f"{formatted_plate} - {district}"
                    break

        # Save to file only if valid plate found
        ocr_output_path = "/Users/mithunravi/Desktop/Number-Plate/ocr-notebook/detected_plate.txt"
        with open(ocr_output_path, "a") as f:
            f.write(f"[scaned_img_{count}.jpg]:\n")
            if detected_plate:
                f.write(detected_plate + "\n\n")
            else:
                f.write("No valid TN plate detected\n\n")

        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
