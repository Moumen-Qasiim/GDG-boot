# Snake CV (Hand Tracking)

A cyberpunk-themed Snake game controlled by hand tracking using Mediapipe and Pygame.

## Features
- **Hand Tracking**: Control the snake's direction by moving your hand across the screen.
- **Cyberpunk Aesthetics**: Neon colors, pixel fonts, and particle effects.
- **Dynamic Difficulty**: The snake speeds up as you eat more food.
- **Auto-Model Download**: Automatically downloads the required Mediapipe hand tracking model.

## Controls
- **Move Hand**: Direct the snake (Top, Bottom, Left, Right quadrants).
- **SPACE**: Start Game / Restart.
- **F**: Toggle Fullscreen.
- **ESC**: Exit.

## Requirements
- Python 3.8+
- OpenCV
- Mediapipe
- Pygame
- Numpy

## Installation
```bash
pip install -r requirements.txt
```

## Running the Game
```bash
python main.py
```

## How it Works
The game uses your webcam to track your index finger. The screen is divided into four zones relative to the center. Moving your finger into a zone will set the snake's direction towards that side.
- **Top Zone**: Snake moves UP.
- **Bottom Zone**: Snake moves DOWN.
- **Left Zone**: Snake moves LEFT.
- **Right Zone**: Snake moves RIGHT.
