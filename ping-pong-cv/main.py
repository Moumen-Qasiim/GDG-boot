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

# Game Specs (Fixed Internal Resolution for Scaling)
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 120 
WINNING_SCORE = 3

# High Speed Physics
BALL_START_SPEED = 6 
BALL_MAX_SPEED = 20
BALL_SIZE = 22
PADDLE_WIDTH, PADDLE_HEIGHT = 24, 160
PADDLE_SMOOTHING = 0.7 # Near-instant follow

# Colors
COLOR_BG = (2, 2, 8)
COLOR_GRID = (10, 10, 25)
COLOR_P1 = (0, 255, 255)
COLOR_P2 = (255, 0, 255)
COLOR_BALL = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)

STATE_MENU = 0
STATE_COUNTDOWN = 1
STATE_PLAYING = 2
STATE_VICTORY = 3
STATE_SCORED = 4

# --- Pixel Font ---
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
    'X': [17,10,4,10,17], 'J': [8,16,16,16,15]
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
    """Completely decoupled vision thread for absolute max performance."""
    def __init__(self):
        super().__init__(daemon=True)
        self._ensure_model_exists()
        self.ly, self.ry = None, None
        self.frame = None
        self.running = True
        
        base = python.BaseOptions(model_asset_path=str(MODEL_PATH), delegate=python.BaseOptions.Delegate.GPU)
        
        def callback(result, output_image, timestamp_ms):
            if result.hand_landmarks:
                l, r = None, None
                for lms in result.hand_landmarks:
                    tip = lms[8]
                    if tip.x < 0.5: l = tip.y
                    else: r = tip.y
                self.ly, self.ry = l, r

        options = vision.HandLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=2, min_hand_detection_confidence=0.3, result_callback=callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

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
            time.sleep(0.001)

class Particle:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.color = color
        self.vx, self.vy = random.uniform(-12, 12), random.uniform(-12, 12)
        self.life = 1.0

    def update(self):
        self.x += self.vx; self.y += self.vy
        self.life -= 0.08
        return self.life > 0

    def draw(self, screen):
        s = pygame.Surface((8, 8), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, int(self.life*255)), (4, 4), 4)
        screen.blit(s, (self.x-4, self.y-4))

class Paddle:
    def __init__(self, x, color):
        self.rect = pygame.Rect(x, SCREEN_HEIGHT//2 - PADDLE_HEIGHT//2, PADDLE_WIDTH, PADDLE_HEIGHT)
        self.color = color; self.ty = self.rect.y

    def update(self, ny):
        if ny is not None: self.ty = ny * SCREEN_HEIGHT - (PADDLE_HEIGHT//2)
        self.rect.y += (self.ty - self.rect.y) * PADDLE_SMOOTHING
        self.rect.top = max(0, self.rect.top); self.rect.bottom = min(SCREEN_HEIGHT, self.rect.bottom)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, border_radius=10)
        pygame.draw.rect(screen, COLOR_WHITE, self.rect, width=2, border_radius=10)

class Ball:
    def __init__(self):
        self.rect = pygame.Rect(0, 0, BALL_SIZE, BALL_SIZE)
        self.trail = [] # List of (x, y, time)
        self.reset()

    def reset(self):
        self.rect.center = (SCREEN_WIDTH//2, SCREEN_HEIGHT//2)
        self.dx = (BALL_START_SPEED if random.random() > 0.5 else -BALL_START_SPEED)
        self.dy = random.uniform(-8, 8)
        self.trail = []

    def update(self):
        # Store trail position
        self.trail.insert(0, (self.rect.centerx, self.rect.centery))
        if len(self.trail) > 15: self.trail.pop()
        
        self.rect.x += self.dx; self.rect.y += self.dy
        if self.rect.top <= 0 or self.rect.bottom >= SCREEN_HEIGHT:
            self.dy *= -1; return True
        return False

    def draw(self, screen):
        # Draw Trail Animation
        for i, pos in enumerate(self.trail):
            # Fading effect
            alpha = int(255 * (1 - i / len(self.trail)) * 0.5)
            size = int(BALL_SIZE * (1 - i / (len(self.trail) * 1.5)))
            s = pygame.Surface((size*2, size*2), pygame.SRCALPHA)
            pygame.draw.circle(s, (*COLOR_BALL, alpha), (size, size), size)
            screen.blit(s, (pos[0]-size, pos[1]-size))

        pygame.draw.ellipse(screen, COLOR_BALL, self.rect)
        pygame.draw.ellipse(screen, COLOR_WHITE, self.rect.inflate(-8, -8), width=2)

class PongGame:
    def __init__(self):
        pygame.init()
        # ENABLE AUTO-SCALING FULLSCREEN BY DEFAULT
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
        pygame.display.set_caption("GDG ZU PING PONG")
        self.clock = pygame.time.Clock()
        self.tracker = HandTrackerThread()
        self.tracker.start()
        self.p1 = Paddle(50, COLOR_P1); self.p2 = Paddle(SCREEN_WIDTH-50-PADDLE_WIDTH, COLOR_P2)
        self.ball = Ball(); self.score = [0, 0]
        self.state = STATE_MENU; self.particles = []; self.timer = 0; self.last = ""
        self.is_fullscreen = True
        self.shake_timer = 0

    def run(self):
        while True:
            # 1. Shake Logic
            offset_x, offset_y = 0, 0
            if self.shake_timer > 0:
                self.shake_timer -= 1
                offset_x = random.randint(-5, 5)
                offset_y = random.randint(-5, 5)

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
                    if event.key == pygame.K_SPACE and self.state in [STATE_MENU, STATE_VICTORY]:
                        self.score = [0, 0]; self.state = STATE_COUNTDOWN; self.timer = 180

            keys = pygame.key.get_pressed()
            kb1 = (self.p1.rect.centery - 35)/SCREEN_HEIGHT if keys[pygame.K_w] else (self.p1.rect.centery + 35)/SCREEN_HEIGHT if keys[pygame.K_s] else None
            kb2 = (self.p2.rect.centery - 35)/SCREEN_HEIGHT if keys[pygame.K_UP] else (self.p2.rect.centery + 35)/SCREEN_HEIGHT if keys[pygame.K_DOWN] else None
            
            v1, v2 = self.tracker.ly, self.tracker.ry
            self.p1.update(v1 if v1 is not None else kb1); self.p2.update(v2 if v2 is not None else kb2)

            if self.state == STATE_PLAYING:
                if self.ball.update(): 
                    self.shake_timer = 5
                    for _ in range(12): self.particles.append(Particle(self.ball.rect.centerx, self.ball.rect.centery, COLOR_WHITE))
                
                if self.ball.rect.colliderect(self.p1.rect) and self.ball.dx < 0:
                    self.ball.dx = min(BALL_MAX_SPEED, abs(self.ball.dx)*1.1); self.ball.rect.left = self.p1.rect.right
                    self.shake_timer = 10
                    for _ in range(20): self.particles.append(Particle(self.ball.rect.left, self.ball.rect.centery, COLOR_P1))
                if self.ball.rect.colliderect(self.p2.rect) and self.ball.dx > 0:
                    self.ball.dx = -min(BALL_MAX_SPEED, abs(self.ball.dx)*1.1); self.ball.rect.right = self.p2.rect.left
                    self.shake_timer = 10
                    for _ in range(20): self.particles.append(Particle(self.ball.rect.right, self.ball.rect.centery, COLOR_P2))

                if self.ball.rect.left <= 0 or self.ball.rect.right >= SCREEN_WIDTH:
                    sidx = 1 if self.ball.rect.left <= 0 else 0
                    self.score[sidx] += 1; self.last = f"PLAYER {sidx+1}"
                    if self.score[sidx] >= WINNING_SCORE: self.state = STATE_VICTORY
                    else: self.state = STATE_SCORED; self.timer = 90
            
            elif self.state == STATE_SCORED:
                self.timer -= 1
                if self.timer <= 0: self.ball.reset(); self.state = STATE_COUNTDOWN; self.timer = 120
            elif self.state == STATE_COUNTDOWN:
                self.timer -= 1
                if self.timer <= 0: self.state = STATE_PLAYING

            self.particles = [p for p in self.particles if p.update()]
            self.screen.fill(COLOR_BG)
            for i in range(0, SCREEN_WIDTH, 120): pygame.draw.line(self.screen, COLOR_GRID, (i, 0), (i, SCREEN_HEIGHT))
            
            if self.state == STATE_MENU:
                draw_pixel_text(self.screen, "GDG ZU PING PONG", SCREEN_WIDTH//2, 250, scale=12, center=True)
                draw_pixel_text(self.screen, "SPACE TO START | ESC TO EXIT | F TO TOGGLE FULLSCREEN", SCREEN_WIDTH//2, 420, scale=3, color=COLOR_BALL, center=True)
            elif self.state in [STATE_PLAYING, STATE_COUNTDOWN, STATE_SCORED]:
                draw_pixel_text(self.screen, f"P1:{self.score[0]}", SCREEN_WIDTH//4, 40, scale=6, color=COLOR_P1, center=True)
                draw_pixel_text(self.screen, f"P2:{self.score[1]}", 3*SCREEN_WIDTH//4, 40, scale=6, color=COLOR_P2, center=True)
                self.p1.draw(self.screen); self.p2.draw(self.screen)
                if self.state == STATE_PLAYING: self.ball.draw(self.screen)
                for p in self.particles: p.draw(self.screen)
                if self.state == STATE_COUNTDOWN: draw_pixel_text(self.screen, str(self.timer//60 + 1), SCREEN_WIDTH//2, 300, scale=15, center=True)
                if self.state == STATE_SCORED: draw_pixel_text(self.screen, f"{self.last} SCORED!", SCREEN_WIDTH//2, 300, scale=8, center=True)
            elif self.state == STATE_VICTORY:
                draw_pixel_text(self.screen, f"{self.last} WINS!", SCREEN_WIDTH//2, 300, scale=12, color=COLOR_BALL, center=True)
                draw_pixel_text(self.screen, "SPACE TO REMATCH", SCREEN_WIDTH//2, 450, scale=4, center=True)

            if self.tracker.frame is not None:
                pw, ph = 120, 90
                ps = pygame.surfarray.make_surface(cv2.cvtColor(cv2.resize(self.tracker.frame, (pw, ph)), cv2.COLOR_BGR2RGB).swapaxes(0,1))
                self.screen.blit(ps, (SCREEN_WIDTH//2-pw//2, SCREEN_HEIGHT-ph-15))
            
            pygame.display.flip(); self.clock.tick(FPS)

if __name__ == "__main__":
    PongGame().run()
