import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pygame
import sys
import urllib.request
from pathlib import Path
import numpy as np
import random
import threading
import time

# --- Configuration ---
MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# UI Specs
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# Colors (Cyberpunk Neon)
COLOR_BG = (2, 2, 8)
COLOR_GRID = (15, 15, 35)
COLOR_AI = (255, 0, 255) # Neon Magenta
COLOR_PLAYER = (0, 255, 255) # Neon Cyan
COLOR_WHITE = (255, 255, 255)
COLOR_ROCK = (200, 200, 200)
COLOR_PAPER = (255, 255, 0)
COLOR_SCISSORS = (0, 255, 0)

STATE_MENU = 0
STATE_COUNTDOWN = 1
STATE_RESULT = 2
STATE_MATCH_OVER = 3

# ... (Pixel Font remains same)

# --- Pixel Font ---
# (Keeping the dictionary as is)
PIXEL_CHARS = {
    '0': [31,17,17,17,31], '1': [0,0,31,0,0], '2': [29,21,21,21,23], '3': [21,21,21,21,31],
    '4': [7,4,4,4,31], '5': [23,21,21,21,29], '6': [31,21,21,21,29], '7': [1,1,1,1,31],
    '8': [31,21,21,21,31], '9': [23,21,21,21,31], 'P': [31,5,5,5,2], 'A': [30,5,5,5,30],
    'L': [31,16,16,16,16], 'Y': [7,8,16,8,7], 'E': [31,21,21,21,21], 'R': [31,5,13,21,18],
    'S': [18,21,21,21,9], 'T': [1,1,31,1,1], 'C': [14,17,16,16,14], 'O': [14,17,17,17,14],
    'N': [31,2,4,8,31], 'G': [14,17,21,21,13], 'I': [17,31,17,0,0], 'W': [31,16,14,16,31],
    'U': [31,16,16,16,31], 'M': [31,2,4,2,31], ' ': [0,0,0,0,0], '!': [29,0,0,0,0],
    'K': [31,4,10,17,0], 'B': [31,21,21,21,10], 'D': [31,17,17,17,14], 'Z': [17,25,21,19,17],
    'H': [31,4,4,4,31], 'F': [31,5,5,5,5], 'V': [15,16,16,16,15], 'Q': [14,17,17,9,22],
    'X': [17,10,4,10,17], 'J': [8,16,16,16,15], '-': [4,4,4,4,4], ':': [0,10,0,0,0],
    '.': [0,16,0,0,0]
}

def draw_pixel_text(screen, text, x, y, scale=4, color=COLOR_WHITE, center=False):
    text = str(text).upper()
    tw = len(text) * (scale * 6)
    cx = x - (tw // 2) if center else x
    for char in text:
        if char in PIXEL_CHARS:
            for ci, cd in enumerate(PIXEL_CHARS[char]):
                for ri in range(5):
                    if (cd >> ri) & 1:
                        pygame.draw.rect(screen, color, (cx + ci*scale, y + ri*scale, scale, scale))
        cx += scale * 6

class HandTrackerThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self._ensure_model_exists()
        self.fingers = [0, 0, 0, 0, 0] # Thumb, Index, Middle, Ring, Pinky
        self.frame = None
        self.running = True
        
        base = python.BaseOptions(model_asset_path=str(MODEL_PATH), delegate=python.BaseOptions.Delegate.GPU)
        
        def callback(result, output_image, timestamp_ms):
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                f = [0]*5
                if abs(lms[4].x - lms[2].x) > abs(lms[3].x - lms[2].x): f[0] = 1
                if lms[8].y < lms[6].y: f[1] = 1
                if lms[12].y < lms[10].y: f[2] = 1
                if lms[16].y < lms[14].y: f[3] = 1
                if lms[20].y < lms[18].y: f[4] = 1
                self.fingers = f
            else:
                self.fingers = [0, 0, 0, 0, 0]

        options = vision.HandLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1, min_hand_detection_confidence=0.5, result_callback=callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.cap = None
        for i in [0, 1, 2]:
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                self.cap = cap
                break
        
        if not self.cap:
            sys.exit(1)

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def _ensure_model_exists(self):
        if not MODEL_PATH.exists(): urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))

    def run(self):
        while self.running:
            ret, raw = self.cap.read()
            if ret:
                self.frame = cv2.flip(raw, 1)
                rgb = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                self.detector.detect_async(mp_img, int(time.time() * 1000))
            time.sleep(0.01)

class Particle:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.color = color
        self.vx, self.vy = random.uniform(-10, 10), random.uniform(-10, 10)
        self.life = 1.0

    def update(self):
        self.x += self.vx; self.y += self.vy
        self.life -= 0.03
        return self.life > 0

    def draw(self, screen):
        s = pygame.Surface((8, 8), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, int(self.life*255)), (4, 4), 4)
        screen.blit(s, (self.x-4, self.y-4))

class RPSGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
        pygame.display.set_caption("GDG ZU ROCK PAPER SCISSORS")
        self.clock = pygame.time.Clock()
        self.tracker = HandTrackerThread()
        self.tracker.start()
        
        self.state = STATE_MENU
        self.scores = [0, 0] # AI, Player
        self.countdown = 0
        self.ai_move = None
        self.player_move = None
        self.result_text = ""
        self.final_winner = ""
        self.particles = []
        self.is_fullscreen = True

    def get_move_name(self, fingers):
        up_count = sum(fingers)
        if up_count == 0: return "ROCK"
        if up_count >= 4: return "PAPER"
        if fingers[1] == 1 and fingers[2] == 1: return "SCISSORS"
        return "UNKNOWN"

    def run(self):
        while True:
            dt = self.clock.tick(FPS)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: sys.exit()
                    if event.key == pygame.K_f:
                        self.is_fullscreen = not self.is_fullscreen
                        if self.is_fullscreen:
                            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
                        else:
                            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SCALED)
                    if event.key == pygame.K_SPACE:
                        if self.state == STATE_MENU or self.state == STATE_MATCH_OVER:
                            self.scores = [0, 0]
                            self.state = STATE_COUNTDOWN
                            self.countdown = 3.5
                        elif self.state == STATE_RESULT:
                            self.state = STATE_COUNTDOWN
                            self.countdown = 3.5
                            self.ai_move = None
                            self.player_move = None

            self.screen.fill(COLOR_BG)
            for i in range(0, SCREEN_WIDTH, 100): pygame.draw.line(self.screen, COLOR_GRID, (i, 0), (i, SCREEN_HEIGHT))
            for i in range(0, SCREEN_HEIGHT, 100): pygame.draw.line(self.screen, COLOR_GRID, (0, i), (SCREEN_WIDTH, i))

            # UI Panels
            pygame.draw.rect(self.screen, COLOR_AI, (20, 20, SCREEN_WIDTH//2 - 40, SCREEN_HEIGHT - 40), width=3, border_radius=15)
            draw_pixel_text(self.screen, f"AI: {self.scores[0]}", 40, 40, scale=6, color=COLOR_AI)
            
            pygame.draw.rect(self.screen, COLOR_PLAYER, (SCREEN_WIDTH//2 + 20, 20, SCREEN_WIDTH//2 - 40, SCREEN_HEIGHT - 40), width=3, border_radius=15)
            draw_pixel_text(self.screen, f"PLAYER: {self.scores[1]}", SCREEN_WIDTH//2 + 40, 40, scale=6, color=COLOR_PLAYER)

            if self.tracker.frame is not None:
                pw, ph = 480, 360
                img = cv2.resize(self.tracker.frame, (pw, ph))
                ps = pygame.surfarray.make_surface(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).swapaxes(0,1))
                self.screen.blit(ps, (SCREEN_WIDTH//2 + (SCREEN_WIDTH//4 - pw//2), SCREEN_HEIGHT//2 - ph//2))

            if self.state == STATE_COUNTDOWN:
                self.countdown -= 1/FPS
                if self.countdown > 0.5:
                    num = int(self.countdown)
                    draw_pixel_text(self.screen, str(num) if num > 0 else "GO!", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50, scale=20, center=True)
                else:
                    self.player_move = self.get_move_name(self.tracker.fingers)
                    self.ai_move = random.choice(["ROCK", "PAPER", "SCISSORS"])
                    self.state = STATE_RESULT
                    
                    if self.player_move == self.ai_move:
                        self.result_text = "DRAW!"
                    elif (self.player_move == "ROCK" and self.ai_move == "SCISSORS") or \
                         (self.player_move == "PAPER" and self.ai_move == "ROCK") or \
                         (self.player_move == "SCISSORS" and self.ai_move == "PAPER"):
                        self.result_text = "POINT PLAYER!"
                        self.scores[1] += 1
                        for _ in range(30): self.particles.append(Particle(SCREEN_WIDTH*0.75, SCREEN_HEIGHT//2, COLOR_PLAYER))
                    else:
                        if self.player_move == "UNKNOWN":
                            self.result_text = "GIVE A GESTURE!"
                        else:
                            self.result_text = "POINT AI!"
                            self.scores[0] += 1
                            for _ in range(30): self.particles.append(Particle(SCREEN_WIDTH*0.25, SCREEN_HEIGHT//2, COLOR_AI))
                    
                    # Check for match over
                    if self.scores[0] >= 3:
                        self.state = STATE_MATCH_OVER
                        self.final_winner = "AI"
                    elif self.scores[1] >= 3:
                        self.state = STATE_MATCH_OVER
                        self.final_winner = "PLAYER"

            elif self.state == STATE_RESULT:
                draw_pixel_text(self.screen, self.ai_move, SCREEN_WIDTH//4, SCREEN_HEIGHT//2, scale=12, color=COLOR_AI, center=True)
                draw_pixel_text(self.screen, self.player_move, SCREEN_WIDTH*0.75, SCREEN_HEIGHT//2, scale=12, color=COLOR_PLAYER, center=True)
                draw_pixel_text(self.screen, self.result_text, SCREEN_WIDTH//2, SCREEN_HEIGHT - 150, scale=10, center=True)
                draw_pixel_text(self.screen, "SPACE FOR NEXT ROUND", SCREEN_WIDTH//2, SCREEN_HEIGHT - 60, scale=4, color=COLOR_GRID, center=True)

            elif self.state == STATE_MATCH_OVER:
                draw_pixel_text(self.screen, f"{self.final_winner} WINS MATCH!", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 50, scale=12, color=COLOR_WHITE, center=True)
                draw_pixel_text(self.screen, "SPACE TO RESTART MATCH", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 50, scale=5, color=COLOR_PLAYER, center=True)
                for _ in range(5):
                    c = COLOR_AI if self.final_winner == "AI" else COLOR_PLAYER
                    self.particles.append(Particle(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), c))

            elif self.state == STATE_MENU:
                draw_pixel_text(self.screen, "ROCK PAPER SCISSORS CV", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 - 100, scale=10, center=True)
                draw_pixel_text(self.screen, "FIRST TO 3 WINS!", SCREEN_WIDTH//2, SCREEN_HEIGHT//2, scale=6, color=COLOR_WHITE, center=True)
                draw_pixel_text(self.screen, "SPACE TO START", SCREEN_WIDTH//2, SCREEN_HEIGHT//2 + 100, scale=5, color=COLOR_PLAYER, center=True)

            self.particles = [p for p in self.particles if p.update()]
            for p in self.particles: p.draw(self.screen)

            pygame.display.flip()

if __name__ == "__main__":
    RPSGame().run()
