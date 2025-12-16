# import cv2
# from ultralytics import YOLO
# import numpy as np

# # =======================
# # ⚙️ CONFIGURATION
# # =======================
# # 1. Standard Model (Detects People)
# MODEL_PERSON_PATH = "yolo11n.pt" 

# # 2. Your Custom Model (Detects ID Cards only)
# MODEL_CARD_PATH = "logo_id/yolov11_run/weights/best.pt" # <--- UPDATE THIS PATH

# CONF_PERSON = 0.5
# CONF_CARD = 0.4

# # Tracking parameters
# IOU_THRESHOLD = 0.3  # How much overlap to consider same person
# MAX_FRAMES_LOST = 150  # Keep authorization for ~5 seconds at 30fps

# # Load Both Models
# print("Loading Person Detector...")
# person_model = YOLO(MODEL_PERSON_PATH)

# print("Loading ID Card Detector...")
# card_model = YOLO(MODEL_CARD_PATH)

# # =======================
# # 📊 TRACKING STATE
# # =======================
# authorized_persons = {}  # {person_id: {'box': [x1,y1,x2,y2], 'frames_lost': 0}}
# next_person_id = 0

# # =======================
# # 🔧 HELPER FUNCTIONS
# # =======================
# def calculate_iou(box1, box2):
#     """Calculate Intersection over Union between two boxes"""
#     x1_1, y1_1, x2_1, y2_1 = box1
#     x1_2, y1_2, x2_2, y2_2 = box2
    
#     # Intersection area
#     x1_i = max(x1_1, x1_2)
#     y1_i = max(y1_1, y1_2)
#     x2_i = min(x2_1, x2_2)
#     y2_i = min(y2_1, y2_2)
    
#     if x2_i < x1_i or y2_i < y1_i:
#         return 0.0
    
#     intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
#     # Union area
#     area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
#     area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
#     union = area1 + area2 - intersection
    
#     return intersection / union if union > 0 else 0.0

# # =======================
# # 🧠 LOGIC FUNCTION
# # =======================
# def check_compliance(frame):
#     global authorized_persons, next_person_id
#     # 1. Run Person Detection
#     person_results = person_model(frame, classes=[0], verbose=False) # Class 0 is Person in COCO
    
#     # 2. Run ID Card Detection
#     card_results = card_model(frame, verbose=False)

#     persons = []
#     ids = []

#     # Extract Person Boxes
#     for r in person_results:
#         for box in r.boxes:
#             if box.conf[0] > CONF_PERSON:
#                 persons.append(box.xyxy[0].cpu().numpy().astype(int))

#     # Extract ID Card Boxes
#     for r in card_results:
#         for box in r.boxes:
#             if box.conf[0] > CONF_CARD:
#                 ids.append(box.xyxy[0].cpu().numpy().astype(int))
#                 # Debug: Draw Cyan box for ID
#                 x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
#                 cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)

#     # 3. Track and match persons with authorized list
#     current_frame_persons = set()
#     matched_authorized = set()
    
#     for person_box in persons:
#         px1, py1, px2, py2 = person_box
#         has_id_now = False
#         best_match_id = None
#         best_iou = 0
        
#         # Check if any ID card is inside this person (new authorization)
#         for cx1, cy1, cx2, cy2 in ids:
#             c_center_x = (cx1 + cx2) // 2
#             c_center_y = (cy1 + cy2) // 2
            
#             if (px1 < c_center_x < px2) and (py1 < c_center_y < py2):
#                 has_id_now = True
#                 break
        
#         # Check if this person matches any previously authorized person
#         for person_id, person_data in authorized_persons.items():
#             iou = calculate_iou(person_box, person_data['box'])
#             if iou > IOU_THRESHOLD and iou > best_iou:
#                 best_iou = iou
#                 best_match_id = person_id
        
#         # Determine authorization status
#         if best_match_id is not None:
#             # Matched with existing authorized person
#             authorized_persons[best_match_id]['box'] = person_box
#             authorized_persons[best_match_id]['frames_lost'] = 0
#             matched_authorized.add(best_match_id)
#             current_frame_persons.add(best_match_id)
            
#             color = (0, 255, 0)  # Green
#             label = f"AUTHORIZED #{best_match_id}"
            
#         elif has_id_now:
#             # New person with ID - authorize them
#             authorized_persons[next_person_id] = {
#                 'box': person_box,
#                 'frames_lost': 0
#             }
#             current_frame_persons.add(next_person_id)
            
#             color = (0, 255, 0)  # Green
#             label = f"AUTHORIZED #{next_person_id}"
#             next_person_id += 1
            
#         else:
#             # Not authorized
#             color = (0, 0, 255)  # Red
#             label = "NO ID"
        
#         # Draw the box and label
#         cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)
#         cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
#     # 4. Update frames_lost for authorized persons not seen this frame
#     persons_to_remove = []
#     for person_id in authorized_persons:
#         if person_id not in matched_authorized:
#             authorized_persons[person_id]['frames_lost'] += 1
#             if authorized_persons[person_id]['frames_lost'] > MAX_FRAMES_LOST:
#                 persons_to_remove.append(person_id)
    
#     # Remove persons who have been lost for too long
#     for person_id in persons_to_remove:
#         del authorized_persons[person_id]

#     return frame

# # =======================
# # 🚀 RUN CAMERA
# # =======================
# # cap = cv2.VideoCapture("rtsp://admin:deepminD1@192.168.0.2:554/Streaming/Channels/101")
# cap = cv2.VideoCapture(0)

# cap.set(3, 1280) # Width
# cap.set(4, 720)  # Height

# while True:
#     ret, frame = cap.read()
#     if not ret: break

#     frame = check_compliance(frame)

#     cv2.imshow("Hybrid System", frame)
#     if cv2.waitKey(1) == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()



















# Logo with id

import cv2
from ultralytics import YOLO
import numpy as np

# =======================
# ⚙️ CONFIGURATION
# =======================
# 1. Standard Model (Detects People)
MODEL_PERSON_PATH = "yolo11n.pt" 

# 2. Your Custom Model (Detects ID Cards and Logo)
MODEL_CARD_PATH = "logo_id/yolov11_run/weights/best.pt"

# Class Indices (Must match your data.yaml: ['ID', 'Logo', 'Person'])
CLS_ID = 0
CLS_LOGO = 1
CLS_PERSON = 2

CONF_PERSON = 0.5
CONF_CARD = 0.25

# Tracking parameters
IOU_THRESHOLD = 0.3  # How much overlap to consider same person
MAX_FRAMES_LOST = 150  # Keep authorization for ~5 seconds at 30fps

# Load Both Models
print("Loading Person Detector...")
person_model = YOLO(MODEL_PERSON_PATH)

print("Loading ID Card Detector...")
card_model = YOLO(MODEL_CARD_PATH)

# =======================
# 📊 TRACKING STATE
# =======================
authorized_persons = {}  # {person_id: {'box': [x1,y1,x2,y2], 'frames_lost': 0}}
next_person_id = 0

# =======================
# 🔧 HELPER FUNCTIONS
# =======================
def calculate_iou(box1, box2):
    """Calculate Intersection over Union between two boxes"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Intersection area
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i < x1_i or y2_i < y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Union area
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def boxes_overlap(box1, box2):
    """Check if box1 center is inside box2"""
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # Calculate center of box1
    center_x = (x1_1 + x2_1) / 2
    center_y = (y1_1 + y2_1) / 2
    
    # Check if center is inside box2
    return (x1_2 < center_x < x2_2) and (y1_2 < center_y < y2_2)

# =======================
# 🧠 LOGIC FUNCTION
# =======================
def check_compliance(frame):
    global authorized_persons, next_person_id
    
    # 1. Run Person Detection
    person_results = person_model(frame, classes=[0], verbose=False, conf=CONF_PERSON)
    
    # 2. Run ID Card and Logo Detection
    card_results = card_model(frame, verbose=False, conf=CONF_CARD)

    persons = []
    ids = []
    logos = []

    # Extract Person Boxes
    for r in person_results:
        for box in r.boxes:
            persons.append(box.xyxy[0].cpu().numpy().astype(int))

    # Extract ID Card and Logo Boxes
    for r in card_results:
        for box in r.boxes:
            cls = int(box.cls[0])
            coords = box.xyxy[0].cpu().numpy().astype(int)
            
            if cls == CLS_ID:
                ids.append(coords)
                # Debug: Draw Yellow box for ID
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(frame, "ID", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
            elif cls == CLS_LOGO:
                logos.append(coords)
                # Debug: Draw Cyan box for Logo
                x1, y1, x2, y2 = coords
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.putText(frame, "LOGO", (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)

    # 3. Track and match persons with authorized list
    current_frame_persons = set()
    matched_authorized = set()
    
    for person_box in persons:
        px1, py1, px2, py2 = person_box
        has_id_with_logo = False
        best_match_id = None
        best_iou = 0
        
        # Check if this person has an ID card with logo
        for id_box in ids:
            if boxes_overlap(id_box, person_box):
                # ID card found on person, now check for logo on that ID
                for logo_box in logos:
                    if boxes_overlap(logo_box, id_box):
                        has_id_with_logo = True
                        break
                if has_id_with_logo:
                    break
        
        # Check if this person matches any previously authorized person
        for person_id, person_data in authorized_persons.items():
            iou = calculate_iou(person_box, person_data['box'])
            if iou > IOU_THRESHOLD and iou > best_iou:
                best_iou = iou
                best_match_id = person_id
        
        # Determine authorization status
        if best_match_id is not None:
            # Matched with existing authorized person
            authorized_persons[best_match_id]['box'] = person_box
            authorized_persons[best_match_id]['frames_lost'] = 0
            matched_authorized.add(best_match_id)
            current_frame_persons.add(best_match_id)
            
            color = (0, 255, 0)  # Green
            label = f"AUTHORIZED #{best_match_id}"
            
        elif has_id_with_logo:
            # New person with ID and Logo - authorize them
            authorized_persons[next_person_id] = {
                'box': person_box,
                'frames_lost': 0
            }
            current_frame_persons.add(next_person_id)
            
            color = (0, 255, 0)  # Green
            label = f"AUTHORIZED #{next_person_id}"
            next_person_id += 1
            
        else:
            # Not authorized
            color = (0, 0, 255)  # Red
            label = "UNAUTHORIZED"
        
        # Draw the box and label
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 3)
        cv2.putText(frame, label, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    
    # 4. Update frames_lost for authorized persons not seen this frame
    persons_to_remove = []
    for person_id in authorized_persons:
        if person_id not in matched_authorized:
            authorized_persons[person_id]['frames_lost'] += 1
            if authorized_persons[person_id]['frames_lost'] > MAX_FRAMES_LOST:
                persons_to_remove.append(person_id)
    
    # Remove persons who have been lost for too long
    for person_id in persons_to_remove:
        del authorized_persons[person_id]
    
    # Add detection count
    debug_text = f"Person: {len(persons)} | ID: {len(ids)} | Logo: {len(logos)}"
    cv2.putText(frame, debug_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return frame

# =======================
# 🚀 MAIN
# =======================
def main():
    # Open Video Source
    cap = cv2.VideoCapture("rtsp://admin:deepminD1@192.168.0.2:554/Streaming/Channels/101")
    # cap = cv2.VideoCapture(0)
    
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = check_compliance(frame)

        cv2.imshow("Hybrid System - ID Authorization", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
