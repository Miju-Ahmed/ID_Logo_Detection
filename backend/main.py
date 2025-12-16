from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
model = YOLO("best.pt")

# Classes
CLS_ID = 0    # Check your data.yaml to confirm these indices!
CLS_LOGO = 1
CLS_PERSON = 2

def is_box_inside(inner, outer):
    """Check if inner box is >50% inside outer box"""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    if inter_x2 < inter_x1 or inter_y2 < inter_y1: return False

    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    inner_area = (ix2 - ix1) * (iy2 - iy1)
    
    return (inter_area / inner_area) > 0.5

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    # 1. Read Image
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # 2. Run Inference
    # reduced confidence to 0.25 to catch more items
    results = model(img, conf=0.25)[0] 

    persons = []
    ids = []
    logos = []

    # 3. Sort Detections
    for box in results.boxes:
        coords = box.xyxy[0].tolist()
        cls = int(box.cls[0])
        
        # DEBUG: Print what we found to the terminal
        print(f"Found Class {cls} at {coords}")

        if cls == CLS_PERSON: persons.append(coords)
        elif cls == CLS_ID: ids.append(coords)
        elif cls == CLS_LOGO: logos.append(coords)

    detections = []

    # 4. LOGIC: Process Persons (Red/Green)
    for p_box in persons:
        status = "UNAUTHORIZED"
        color = "red"
        
        # Is there an ID inside this person?
        has_id = False
        valid_id_box = None
        for i_box in ids:
            if is_box_inside(i_box, p_box):
                has_id = True
                valid_id_box = i_box
                break
        
        # Is there a Logo inside that ID?
        if has_id and valid_id_box:
            for l_box in logos:
                if is_box_inside(l_box, valid_id_box):
                    status = "AUTHORIZED"
                    color = "green"
                    break

        detections.append({
            "box": p_box,
            "status": status,
            "color": color
        })

    # 5. VISUALIZATION FIX: Add IDs and Logos separately
    # This ensures you see them even if the Person isn't detected
    for i_box in ids:
        detections.append({
            "box": i_box, 
            "status": "ID CARD", 
            "color": "yellow" # Use a distinct color for IDs
        })
        
    for l_box in logos:
        detections.append({
            "box": l_box, 
            "status": "LOGO", 
            "color": "blue" # Use a distinct color for Logos
        })

    print(f"Total Detections sent to React: {len(detections)}")
    return {"detections": detections}