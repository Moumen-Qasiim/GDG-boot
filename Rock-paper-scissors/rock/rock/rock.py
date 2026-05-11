import random
import cv2
import time
import sys
import os
import numpy as np

# Add root to path to import cv_utils
# The path is deep: Rock-paper-scissors/rock/rock/rock.py
# Root is 3 levels up
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))
import cv_utils

# Try to find a working camera
cap = cv_utils.get_camera()
if not cap:
    print("Error: Could not open any camera.")
    sys.exit(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

tracker = cv_utils.HandTracker(max_hands=1, detection_con=0.5)

timer = 0
stateResult = False
startGame = False
scores = [0, 0]  

def overlay_png(img_back, img_front, pos):
    x, y = pos
    h, w, c = img_front.shape
    if y + h > img_back.shape[0] or x + w > img_back.shape[1]:
        return img_back # Out of bounds
    
    if c == 4:
        overlay = img_front[:, :, :3]
        mask = img_front[:, :, 3] / 255.0
        for c_idx in range(3):
            img_back[y:y+h, x:x+w, c_idx] = img_back[y:y+h, x:x+w, c_idx] * (1 - mask) + overlay[:, :, c_idx] * mask
    else:
        img_back[y:y+h, x:x+w] = img_front
    return img_back

while True:
    imgBG = cv2.imread("Resources/BG.png")
    if imgBG is None:
        imgBG = np.zeros((720, 1280, 3), dtype=np.uint8)
        cv2.putText(imgBG, "Background Missing", (400, 360), cv2.FONT_HERSHEY_PLAIN, 3, (255, 255, 255), 2)

    success, img = cap.read()
    if not success:
        continue

    imgScaled = cv2.resize(img, (400, 420)) # Match the region in BG
    
    # Process hands
    tracker.process_frame(imgScaled)
    hands = tracker.get_hands(400, 420)
    
    if hands:
        tracker.draw_landmarks(imgScaled, hands[0])

    if startGame:
        if not stateResult:
            timer = time.time() - initialTime
            # Display countdown 3, 2, 1
            countdown = 3 - int(timer)
            cv2.putText(imgBG, str(max(0, countdown)), (605, 435), cv2.FONT_HERSHEY_PLAIN, 6, (255, 0, 255), 4)

            if timer > 3:
                stateResult = True
                timer = 0

                if hands:
                    playerMove = None
                    hand = hands[0]
                    fingers = tracker.fingers_up(hand)
                    # Mapping fingers to moves:
                    # [0,0,0,0,0] -> Rock (1)
                    # [1,1,1,1,1] -> Paper (2)
                    # [0,1,1,0,0] -> Scissors (3)
                    if fingers == [0, 0, 0, 0, 0]:
                        playerMove = 1
                    elif fingers == [1, 1, 1, 1, 1]:
                        playerMove = 2
                    elif fingers == [0, 1, 1, 0, 0]:
                        playerMove = 3

                    randomNumber = random.randint(1, 3)
                    imgAI = cv2.imread(f'Resources/{randomNumber}.png', cv2.IMREAD_UNCHANGED)
                    if imgAI is not None:
                        imgBG = overlay_png(imgBG, imgAI, (149, 310))

                    # Determine winner
                    if playerMove:
                        if (playerMove == 1 and randomNumber == 3) or \
                        (playerMove == 2 and randomNumber == 1) or \
                        (playerMove == 3 and randomNumber == 2):
                            scores[1] += 1 # AI wins or Player wins? Original code: scores[1]+=1
                        elif (playerMove == 3 and randomNumber == 1) or \
                            (playerMove == 1 and randomNumber == 2) or \
                            (playerMove == 2 and randomNumber == 3):
                            scores[0] += 1

    # Place scaled camera feed into background
    # Original coordinates: 234:654 (height 420), 795:1195 (width 400)
    try:
        imgBG[234:654, 795:1195] = imgScaled
    except:
        pass

    if stateResult:
        if 'imgAI' in locals() and imgAI is not None:
            imgBG = overlay_png(imgBG, imgAI, (149, 310))

    cv2.putText(imgBG, str(scores[0]), (410, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)
    cv2.putText(imgBG, str(scores[1]), (1112, 215), cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 6)

    cv2.imshow("Rock Paper Scissors", imgBG)

    key = cv2.waitKey(1)
    if key == ord('s'):
        startGame = True
        initialTime = time.time()
        stateResult = False
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()