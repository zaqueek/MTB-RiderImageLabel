# Code made by zaqueek on GitHub.
# I have a patreon if you want to support me! :D
# Put your photos in the "Input" folder.
# After you run everything, copies of your photos with the names will end up in an output folder.
# Currently, everything is formatted for the Roots&Rain naming reqirements.

#Lmk if theres any issues.


import os
import cv2
import easyocr
from ultralytics import YOLO

# Load trained YOLO model
MODEL_PATH = "runs/detect/train/weights/best.pt"
plate_model = YOLO(MODEL_PATH)

# OCR reader
ocr = easyocr.Reader(['en'])

# Input/output folders
INPUT_FOLDER = "Input"          # put race photos here
OUTPUT_FOLDER = "renamed_photos"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for file in os.listdir(INPUT_FOLDER):
    if file.lower().endswith(('.jpg', '.png', '.jpeg')):
        img_path = os.path.join(INPUT_FOLDER, file)
        img = cv2.imread(img_path)

        # Detect plates
        results = plate_model(img)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        rider_number = "UNKNOWN"

        if len(boxes) > 0:
            # Use the first detected plate
            x1, y1, x2, y2 = map(int, boxes[0])
            plate_crop = img[y1:y2, x1:x2]

            # OCR
            text_results = ocr.readtext(plate_crop)
            text = " ".join([t[1] for t in text_results])
            digits = "".join([c for c in text if c.isdigit()])

            if digits:
                rider_number = digits

        # New filename
        ext = os.path.splitext(file)[1]
        new_name = f"__{rider_number}__{ext}"
        new_path = os.path.join(OUTPUT_FOLDER, new_name)

        # Save a copy
        cv2.imwrite(new_path, img)
        print(f"✅ {file} -> {new_name}")