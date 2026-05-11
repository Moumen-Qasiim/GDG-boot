import math
import random
import cv2
import numpy as np
import sys
import os

# Add root to path to import cv_utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import cv_utils

# Try to find a working camera
cap = cv_utils.get_camera()
if not cap:
    print("Error: Could not open any camera.")
    sys.exit(1)

cap.set(3, 1280)
cap.set(4, 720)

tracker = cv_utils.HandTracker(max_hands=1, detection_con=0.8)

class SnakeGameClass:
    def __init__(self, pathFood):
        self.points = []  # all points of the snake
        self.lengths = []  # distance between each point
        self.currentLength = 0  # total length of the snake
        self.allowedLength = 150  # total allowed Length
        self.previousHead = 0, 0  # previous head point

        self.imgFood = cv2.imread(pathFood, cv2.IMREAD_UNCHANGED)
        if self.imgFood is None:
            # Create a simple red circle as fallback if Donut.png is missing
            self.imgFood = np.zeros((50, 50, 4), dtype=np.uint8)
            cv2.circle(self.imgFood, (25, 25), 20, (0, 0, 255, 255), -1)
            
        self.hFood, self.wFood, _ = self.imgFood.shape
        self.foodPoint = 0, 0
        self.randomFoodLocation()

        self.score = 0
        self.gameOver = False

    def randomFoodLocation(self):
        self.foodPoint = random.randint(100, 1000), random.randint(100, 600)

    def update(self, imgMain, currentHead):
        if self.gameOver:
            cv2.putText(imgMain, "Game Over", (300, 400), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 255), 5)
            cv2.putText(imgMain, f'Your Score: {self.score}', (300, 550), cv2.FONT_HERSHEY_PLAIN, 7, (255, 0, 255), 5)
        else:
            px, py = self.previousHead
            cx, cy = currentHead

            self.points.append([cx, cy])
            distance = math.hypot(cx - px, cy - py)
            self.lengths.append(distance)
            self.currentLength += distance
            self.previousHead = cx, cy

            # Length Reduction
            if self.currentLength > self.allowedLength:
                for i, length in enumerate(self.lengths):
                    self.currentLength -= length
                    self.lengths.pop(i)
                    self.points.pop(i)
                    if self.currentLength < self.allowedLength:
                        break

            # Check if snake ate the Food
            rx, ry = self.foodPoint
            if rx - self.wFood // 2 < cx < rx + self.wFood // 2 and \
                    ry - self.hFood // 2 < cy < ry + self.hFood // 2:
                self.randomFoodLocation()
                self.allowedLength += 50
                self.score += 1

            # Draw Snake
            if self.points:
                for i, point in enumerate(self.points):
                    if i != 0:
                        cv2.line(imgMain, tuple(self.points[i - 1]), tuple(self.points[i]), (0, 0, 255), 20)
                cv2.circle(imgMain, tuple(self.points[-1]), 20, (0, 255, 0), cv2.FILLED)

            # Draw Food
            # Simplified overlay since we removed cvzone dependency
            y1, y2 = ry - self.hFood // 2, ry + self.hFood // 2
            x1, x2 = rx - self.wFood // 2, rx + self.wFood // 2
            
            if 0 <= y1 < 720 and 0 <= y2 < 720 and 0 <= x1 < 1280 and 0 <= x2 < 1280:
                overlay = self.imgFood[:, :, :3]
                mask = self.imgFood[:, :, 3] / 255.0
                for c in range(3):
                    imgMain[y1:y2, x1:x2, c] = imgMain[y1:y2, x1:x2, c] * (1 - mask) + overlay[:, :, c] * mask

            cv2.putText(imgMain, f'Score: {self.score}', (50, 80), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)

            # Check for Collision
            if len(self.points) > 5:
                pts = np.array(self.points[:-2], np.int32)
                pts = pts.reshape((-1, 1, 2))
                minDist = cv2.pointPolygonTest(pts, (cx, cy), True)

                if -1 <= minDist <= 1:
                    self.gameOver = True
                    self.points = []
                    self.lengths = []
                    self.currentLength = 0
                    self.allowedLength = 150
                    self.previousHead = 0, 0
                    self.randomFoodLocation()

        return imgMain

game = SnakeGameClass("Donut.png")

while True:
    success, img = cap.read()
    if not success: continue
    img = cv2.flip(img, 1)
    
    tracker.process_frame(img)
    hands = tracker.get_hands(1280, 720)

    if hands:
        tracker.draw_landmarks(img, hands[0])
        lmList = hands[0]['lmList']
        pointIndex = lmList[8][0:2]
        img = game.update(img, pointIndex)
        
    cv2.imshow("Snake Game", img)
    key = cv2.waitKey(1)
    if key == ord('r'):
        game.gameOver = False
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()