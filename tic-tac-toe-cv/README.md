# Google Development Group ZU Tic-Tac-Toe

A state-of-the-art, 2-player Tic-Tac-Toe game that utilizes **Computer Vision** and **Artificial Intelligence** to turn your hand into a gaming controller. Built exclusively for the **GDG ZU** event.

## ✨ Key Features
- **Gestural Control (Pinch-to-Place)**: No mouse or keyboard required. Move your hand in front of the camera to navigate and "pinch" your thumb and index finger together to mark your spot.
- **Async AI Pipeline**: Leverages MediaPipe's **Hand Landmarker Tasks API** in a dedicated system thread. This prevents AI inference from blocking the UI, ensuring a consistent 60+ FPS experience.
- **Cyberpunk Visuals**: High-contrast neon aesthetics with:
  - Pulsing neon X and O marks.
  - Particle-based placement effects.
  - Screen-shake and fireworks on victory.
- **Real-Time Feedback**: Circular cursor highlights the current cell and changes color when a "Pinch" is detected.
- **Turn & Winner Logic**: Integrated UI indicators for current player turns and automated win/draw detection.

## 🚀 Installation & Setup

1. **Clone & Navigate**:
   ```bash
   cd tic-tac-toe-cv
   ```

2. **Environment Setup**:
   It is recommended to use a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to Play

### Launching the Game
For standard systems:
```bash
python main.py
```

**For NVIDIA RTX GPU Acceleration (Recommended)**:
On Linux systems with Prime Offload, use this command to force the AI to run on your dedicated NVIDIA GPU:
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python main.py
```

### Controls
| Action | Gesture / Key |
| :--- | :--- |
| **Move Cursor** | Move Hand (Index/Thumb midpoint) |
| **Place Mark** | **Pinch** (Bring Thumb and Index together) |
| **Start Match** | `SPACE` |
| **Reset / Rematch** | `SPACE` |
| **Toggle Fullscreen** | `F` |
| **Exit** | `ESC` |

## 🛠️ Technical Stack
- **AI Core**: [MediaPipe Tasks API](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) (GPU Delegate enabled)
- **Engine**: [Pygame](https://www.pygame.org/) (SCALED mode for automatic high-res scaling)
- **Vision**: [OpenCV](https://opencv.org/) (Zero-latency buffer configuration)
- **Architecture**: Asynchronous Multithreaded Producer-Consumer Pattern

---
Developed for **Google Development Group ZU**. Enjoy the future of interactive gaming! ❌⭕️⚡️
