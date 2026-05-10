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
GRID_SIZE = 450
CELL_SIZE = GRID_SIZE // 3

# Colors (Cyberpunk Neon)
COLOR_BG = (2, 2, 8)
COLOR_GRID = (0, 255, 255)
COLOR_X = (255, 0, 255)
COLOR_O = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_HIGHLIGHT = (50, 50, 100)

STATE_MENU = 0
STATE_PLAYING = 1
STATE_VICTORY = 2

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
    'X': [17,10,4,10,17], 'J': [8,16,16,16,15], '-': [4,4,4,4,4], '|': [31,0,0,0,0]
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
        self.pinched = False
        self.frame = None
        self.running = True
        
        base = python.BaseOptions(model_asset_path=str(MODEL_PATH), delegate=python.BaseOptions.Delegate.GPU)
        
        def callback(result, output_image, timestamp_ms):
            if result.hand_landmarks:
                for lms in result.hand_landmarks:
                    thumb = lms[4]
                    index = lms[8]
                    # Midpoint for cursor
                    self.cursor = ((thumb.x + index.x)/2, (thumb.y + index.y)/2)
                    # Pinch detection (distance between thumb and index)
                    dist = ((thumb.x - index.x)**2 + (thumb.y - index.y)**2)**0.5
                    self.pinched = dist < 0.05
                    break # Track only first hand
            else:
                self.cursor = None
                self.pinched = False

        options = vision.HandLandmarkerOptions(
            base_options=base, running_mode=vision.RunningMode.LIVE_STREAM,
            num_hands=1, min_hand_detection_confidence=0.5, result_callback=callback
        )
        self.detector = vision.HandLandmarker.create_from_options(options)
        self.cap = cv2.VideoCapture(0)

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
        self.vx, self.vy = random.uniform(-5, 5), random.uniform(-5, 5)
        self.life = 1.0

    def update(self):
        self.x += self.vx; self.y += self.vy
        self.life -= 0.05
        return self.life > 0

    def draw(self, screen):
        s = pygame.Surface((6, 6), pygame.SRCALPHA)
        pygame.draw.circle(s, (*self.color, int(self.life*255)), (3, 3), 3)
        screen.blit(s, (self.x-3, self.y-3))

class TicTacToe:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SCALED)
        pygame.display.set_caption("GDG ZU TIC-TAC-TOE")
        self.clock = pygame.time.Clock()
        self.tracker = HandTrackerThread()
        self.tracker.start()
        
        self.reset_game()
        self.particles = []
        self.shake = 0

    def reset_game(self):
        self.board = [None] * 9
        self.turn = 'X'
        self.state = STATE_MENU
        self.winner = None
        self.last_pinch = 0

    def check_winner(self):
        wins = [(0,1,2), (3,4,5), (6,7,8), (0,3,6), (1,4,7), (2,5,8), (0,4,8), (2,4,6)]
        for w in wins:
            if self.board[w[0]] and self.board[w[0]] == self.board[w[1]] == self.board[w[2]]:
                return self.board[w[0]]
        if None not in self.board: return "DRAW"
        return None

    def run(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT: sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE: sys.exit()
                    if event.key == pygame.K_SPACE and self.state != STATE_PLAYING:
                        self.reset_game()
                        self.state = STATE_PLAYING

            self.screen.fill(COLOR_BG)
            
            # Rendering Grid
            grid_rect = pygame.Rect(SCREEN_WIDTH//2 - GRID_SIZE//2, SCREEN_HEIGHT//2 - GRID_SIZE//2, GRID_SIZE, GRID_SIZE)
            
            # Cursor Logic
            cx, cy = -100, -100
            if self.tracker.cursor:
                cx = int(self.tracker.cursor[0] * SCREEN_WIDTH)
                cy = int(self.tracker.cursor[1] * SCREEN_HEIGHT)

            if self.state == STATE_PLAYING:
                # Draw Grid Lines
                for i in range(1, 3):
                    pygame.draw.line(self.screen, COLOR_GRID, (grid_rect.left + i*CELL_SIZE, grid_rect.top), (grid_rect.left + i*CELL_SIZE, grid_rect.bottom), 5)
                    pygame.draw.line(self.screen, COLOR_GRID, (grid_rect.left, grid_rect.top + i*CELL_SIZE), (grid_rect.right, grid_rect.top + i*CELL_SIZE), 5)
                
                # Check hovering cell
                hover_cell = None
                if grid_rect.collidepoint(cx, cy):
                    ix = (cx - grid_rect.left) // CELL_SIZE
                    iy = (cy - grid_rect.top) // CELL_SIZE
                    hover_cell = iy * 3 + ix
                    
                    # Draw Hover Highlight
                    h_rect = pygame.Rect(grid_rect.left + ix*CELL_SIZE, grid_rect.top + iy*CELL_SIZE, CELL_SIZE, CELL_SIZE)
                    pygame.draw.rect(self.screen, COLOR_HIGHLIGHT, h_rect.inflate(-10, -10), border_radius=15)

                    # Handle Placement
                    if self.tracker.pinched and time.time() - self.last_pinch > 0.5:
                        if self.board[hover_cell] is None:
                            self.board[hover_cell] = self.turn
                            self.turn = 'O' if self.turn == 'X' else 'X'
                            self.last_pinch = time.time()
                            # Particles on place
                            for _ in range(15): self.particles.append(Particle(cx, cy, COLOR_WHITE))
                            
                            self.winner = self.check_winner()
                            if self.winner:
                                self.state = STATE_VICTORY
                                self.shake = 20

                # Draw Marks
                for i, mark in enumerate(self.board):
                    if mark:
                        mx = grid_rect.left + (i % 3) * CELL_SIZE + CELL_SIZE // 2
                        my = grid_rect.top + (i // 3) * CELL_SIZE + CELL_SIZE // 2
                        color = COLOR_X if mark == 'X' else COLOR_O
                        draw_pixel_text(self.screen, mark, mx, my - 20, scale=12, color=color, center=True)

            elif self.state == STATE_MENU:
                draw_pixel_text(self.screen, "GOOGLE DEVELOPMENT GROUP", SCREEN_WIDTH//2, 200, scale=6, center=True)
                draw_pixel_text(self.screen, "TIC-TAC-TOE", SCREEN_WIDTH//2, 280, scale=14, center=True)
                draw_pixel_text(self.screen, "PINCH TO PLACE | SPACE TO START", SCREEN_WIDTH//2, 450, scale=4, color=COLOR_GRID, center=True)

            elif self.state == STATE_VICTORY:
                txt = f"{self.winner} WINS!" if self.winner != "DRAW" else "DRAW!"
                draw_pixel_text(self.screen, txt, SCREEN_WIDTH//2, 300, scale=15, color=COLOR_WHITE, center=True)
                draw_pixel_text(self.screen, "SPACE TO PLAY AGAIN", SCREEN_WIDTH//2, 450, scale=4, center=True)
                for _ in range(5): self.particles.append(Particle(random.randint(0, SCREEN_WIDTH), random.randint(0, SCREEN_HEIGHT), random.choice([COLOR_X, COLOR_O, COLOR_GRID])))

            # Particles & Shake
            if self.shake > 0:
                self.shake -= 1
                shake_x = random.randint(-5, 5)
                shake_y = random.randint(-5, 5)
                # Note: Screen shake implementation usually requires blitting the whole screen offset
            
            self.particles = [p for p in self.particles if p.update()]
            for p in self.particles: p.draw(self.screen)

            # Draw Cursor
            if self.tracker.cursor:
                p_color = COLOR_WHITE if not self.tracker.pinched else COLOR_GRID
                pygame.draw.circle(self.screen, p_color, (cx, cy), 15 if not self.tracker.pinched else 25, width=3)
                pygame.draw.circle(self.screen, p_color, (cx, cy), 5)

            pygame.display.flip()
            self.clock.tick(FPS)

if __name__ == "__main__":
    TicTacToe().run()
