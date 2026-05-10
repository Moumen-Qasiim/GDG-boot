# Google Development Group ZU Tic-Tac-Toe

A high-performance, camera-based Tic Tac Toe game built for **GDG ZU**. Control the game with your hands using cutting-edge AI!

## ✨ Features
- **Pinch-to-Place**: Move your hand to hover over a cell and pinch your thumb and index finger to place your mark.
- **AI Hand Tracking**: Powered by MediaPipe Tasks for zero-latency detection.
- **Cyberpunk Aesthetic**: Neon visuals with pulsing effects and particle explosions.
- **120 FPS Target**: Silky smooth rendering decoupled from AI processing.
- **RTX Optimized**: Supports NVIDIA GPU acceleration.

## 🚀 Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Game**:
   ```bash
   # Standard
   python main.py
   
   # For RTX GPUs (NVIDIA)
   __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python main.py
   ```

## 🎮 Controls
- **Move Hand**: Hover over the grid.
- **Pinch (Thumb + Index)**: Place X or O.
- **SPACE**: Start Game / Rematch.
- **ESC**: Exit.

---
Developed for **GDG ZU**. Enjoy the match! ❌⭕️⚡️
