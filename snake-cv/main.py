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
import math

# --- Configuration ---
MODEL_PATH = Path("hand_landmarker.task")
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"

# Game Specs
SCREEN_WIDTH = 1280
SCREEN_HEIGHT = 720
FPS = 60

# Colors (Cyberpunk Neon)
COLOR_BG = (2, 2, 8)
COLOR_GRID = (15, 15, 35)
COLOR_SNAKE = (0, 255, 150)
COLOR_SNAKE_HEAD = (200, 255, 200)
COLOR_FOOD = (255, 0, 150)
COLOR_WHITE = (255, 255, 255)
COLOR_UI = (0, 200, 255)

STATE_MENU = 0
STATE_PLAYING = 1
STATE_GAMEOVER = 2

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
    'X': [17,10,4,10,17], 'J': [8,16,16,16,15], '-': [4,4,4,4,4], '|': [31,0,0,0,0],
    ':': [0,10,0,0,0], '.': [0,16,0,0,0]
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
        self.cursor = None # (x, y) normalized
        self.frame = None
        self.running = True
        
        base = python.BaseOptions(model_asset_path=str(MODEL_PATH), delegate=python.BaseOptions.Delegate.GPU)
        
        def callback(result, output_image, timestamp_ms):
            if result.hand_landmarks:
                lms = result.hand_landmarks[0]
                self.cursor = (lms[8].x, lms[8].y)
            else:
                self.cursor = None

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
            time.sleep(0.01)

class Particle:
    def __init__(self, x, y, color):
        self.x, self.y = x, y
        self.color = color
        self.vx, self.vy = random.uniform(-6, 6), random.uniform(-6, 6)
        self.life = 1.0
        self.size = random.randint(3, 6)

    def update(self):
        self.x += self.vx; self.y += self.vy
        self.life -= 0.04
        return self.life > 0

    def draw(self, screen):
        alpha = int(self.life * 255)
        s = pygame.Surface((self.size*2, self.size*2), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, alpha), (self.size, self.size), self.size)
        screen.blit(s, (self.x - self.size, self.y - self.size))

class SnakeGame:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.FULLSCREEN | pygame.SCALED)
        pygame.display.set_caption("GDG ZU SNAKE CV")
        self.clock = pygame.time.Clock()
        self.tracker = HandTrackerThread()
        self.tracker.start()
        
        self.particles = []
        self.is_fullscreen = True
        
        # Cursor smoothing
        self.smooth_cursor = [0.5, 0.5]
        self.smoothing = 0.15 # Lower is smoother/slower

        self.reset_game()

    def reset_game(self):
        self.head_x = SCREEN_WIDTH // 2
        self.head_y = SCREEN_HEIGHT // 2
        self.angle = 0
        self.speed = 4.0
        self.turn_speed = 0.12 # radians per frame
        
        self.trail = [] # List of (x, y) coordinates
        self.snake_length = 40 # Number of trail points to keep
        self.segments = 5 # Initial segments
        
        self.food = self.spawn_food()
        self.score = 0
        self.state = STATE_MENU
        self.shake = 0

    def spawn_food(self):
        return (random.randint(50, SCREEN_WIDTH - 50), random.randint(50, SCREEN_HEIGHT - 50))

    def update_movement(self):
        if self.tracker.cursor:
            # Smooth cursor position
            self.smooth_cursor[0] += (self.tracker.cursor[0] - self.smooth_cursor[0]) * self.smoothing
            self.smooth_cursor[1] += (self.tracker.cursor[1] - self.smooth_cursor[1]) * self.smoothing
            
            target_x = self.smooth_cursor[0] * SCREEN_WIDTH
            target_y = self.smooth_cursor[1] * SCREEN_HEIGHT
            
            # Calculate angle to target
            dx = target_x - self.head_x
            dy = target_y - self.head_y
            target_angle = math.atan2(dy, dx)
            
            # Smoothly rotate towards target angle
            # Normalize angle difference to [-pi, pi]
            angle_diff = (target_angle - self.angle + math.pi) % (2 * math.pi) - math.pi
            if abs(angle_diff) > 0.01:
                self.angle += np.clip(angle_diff, -self.turn_speed, self.turn_speed)

        # Move head
        self.head_x += self.speed * math.cos(self.angle)
        self.head_y += self.speed * math.sin(self.angle)
        
        # Wall collisions
        if self.head_x < 0 or self.head_x > SCREEN_WIDTH or self.head_y < 0 or self.head_y > SCREEN_HEIGHT:
            self.trigger_gameover()
            return

        # Update trail
        self.trail.insert(0, (self.head_x, self.head_y))
        
        # Max trail length depends on score (segments)
        max_trail = (self.segments + 3) * 15 # roughly 15 points per visual segment
        if len(self.trail) > max_trail:
            self.trail.pop()
            
        # Self-collision (skip the head area)
        if len(self.trail) > 40:
            for i in range(40, len(self.trail), 5):
                dist = math.hypot(self.head_x - self.trail[i][0], self.head_y - self.trail[i][1])
                if dist < 12:
                    self.trigger_gameover()
                    return

        # Food collision
        dist_to_food = math.hypot(self.head_x - self.food[0], self.head_y - self.food[1])
        if dist_to_food < 25:
            self.score += 1
            self.segments += 1
            self.speed = min(8.0, 4.0 + self.score * 0.1)
            self.food = self.spawn_food()
            self.shake = 10
            for _ in range(15):
                self.particles.append(Particle(self.head_x, self.head_y, COLOR_FOOD))

    def trigger_gameover(self):
        self.state = STATE_GAMEOVER
        self.shake = 20
        for _ in range(40):
            self.particles.append(Particle(self.head_x, self.head_y, COLOR_SNAKE))

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
                        if self.state != STATE_PLAYING:
                            self.reset_game()
                            self.state = STATE_PLAYING

            self.screen.fill(COLOR_BG)
            
            # Draw aesthetic grid
            for x in range(0, SCREEN_WIDTH, 100):
                pygame.draw.line(self.screen, COLOR_GRID, (x, 0), (x, SCREEN_HEIGHT))
            for y in range(0, SCREEN_HEIGHT, 100):
                pygame.draw.line(self.screen, COLOR_GRID, (0, y), (SCREEN_WIDTH, y))

            if self.state == STATE_PLAYING:
                self.update_movement()
                
                # Draw Food (Glowy)
                fx, fy = self.food
                pulse = abs(math.sin(time.time() * 5)) * 8
                # Outer glow
                s = pygame.Surface((60, 60), pygame.SRCALPHA)
                pygame.draw.circle(s, (*COLOR_FOOD, 50), (30, 30), 20 + pulse)
                self.screen.blit(s, (fx-30, fy-30))
                pygame.draw.circle(self.screen, COLOR_FOOD, (fx, fy), 12)
                pygame.draw.circle(self.screen, COLOR_WHITE, (fx, fy), 6)

                # Draw Snake Body
                # We draw segments every N trail points
                for i in range(len(self.trail) - 1, 0, -10):
                    pos = self.trail[i]
                    size = max(8, 20 - (i / len(self.trail)) * 10)
                    pygame.draw.circle(self.screen, COLOR_SNAKE, (int(pos[0]), int(pos[1])), int(size))

                # Draw Head
                head_size = 22
                pygame.draw.circle(self.screen, COLOR_SNAKE_HEAD, (int(self.head_x), int(self.head_y)), head_size)
                # Eyes
                eye_off = 10
                eye_angle = 0.5
                ex1 = self.head_x + eye_off * math.cos(self.angle - eye_angle)
                ey1 = self.head_y + eye_off * math.sin(self.angle - eye_angle)
                ex2 = self.head_x + eye_off * math.cos(self.angle + eye_angle)
                ey2 = self.head_y + eye_off * math.sin(self.angle + eye_angle)
                pygame.draw.circle(self.screen, COLOR_BG, (int(ex1), int(ey1)), 5)
                pygame.draw.circle(self.screen, COLOR_BG, (int(ex2), int(ey2)), 5)

                draw_pixel_text(self.screen, f"SCORE:{self.score}", 20, 20, scale=4, color=COLOR_UI)

            elif self.state == STATE_MENU:
                draw_pixel_text(self.screen, "GOOGLE DEVELOPMENT GROUP", SCREEN_WIDTH//2, 200, scale=6, center=True)
                draw_pixel_text(self.screen, "SNAKE CV", SCREEN_WIDTH//2, 280, scale=18, color=COLOR_SNAKE, center=True)
                draw_pixel_text(self.screen, "DIRECT WITH HAND | CIRCLE TO TURN | SPACE TO START", SCREEN_WIDTH//2, 450, scale=3, color=COLOR_UI, center=True)

            elif self.state == STATE_GAMEOVER:
                draw_pixel_text(self.screen, "GAME OVER", SCREEN_WIDTH//2, 280, scale=18, color=COLOR_FOOD, center=True)
                draw_pixel_text(self.screen, f"FINAL SCORE: {self.score}", SCREEN_WIDTH//2, 400, scale=6, center=True)
                draw_pixel_text(self.screen, "SPACE TO RESTART", SCREEN_WIDTH//2, 500, scale=4, color=COLOR_UI, center=True)

            # Particles
            self.particles = [p for p in self.particles if p.update()]
            for p in self.particles: p.draw(self.screen)

            # Camera Preview
            if self.tracker.frame is not None:
                pw, ph = 160, 120
                ps = pygame.surfarray.make_surface(cv2.cvtColor(cv2.resize(self.tracker.frame, (pw, ph)), cv2.COLOR_BGR2RGB).swapaxes(0,1))
                self.screen.blit(ps, (SCREEN_WIDTH - pw - 20, SCREEN_HEIGHT - ph - 20))
                pygame.draw.rect(self.screen, COLOR_UI, (SCREEN_WIDTH - pw - 22, SCREEN_HEIGHT - ph - 22, pw + 4, ph + 4), width=2)
                
                if self.tracker.cursor:
                    cx, cy = self.tracker.cursor
                    cv_x = SCREEN_WIDTH - pw - 20 + int(cx * pw)
                    cv_y = SCREEN_HEIGHT - ph - 20 + int(cy * ph)
                    pygame.draw.circle(self.screen, COLOR_SNAKE, (cv_x, cv_y), 5)

            pygame.display.flip()

if __name__ == "__main__":
    SnakeGame().run()
