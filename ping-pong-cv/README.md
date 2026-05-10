# GDG ZU PING PONG

A high-performance, 2-player, camera-based Ping Pong game built for the Google Development Group (GDG). This project utilizes MediaPipe's modern Tasks API for real-time hand tracking and Pygame for an ultra-smooth, arcade-style gaming experience.

## ✨ Features
- **AI Hand Tracking**: Control your paddle using your index finger via your webcam.
- **Multithreaded Architecture**: Vision processing runs in a separate thread to ensure a locked 120 FPS rendering.
- **RTX Optimized**: Supports NVIDIA RTX GPU acceleration via asynchronous live-stream inference.
- **Premium Animations**:
  - **Dynamic Trails**: Motion-blurred trails that follow the ball's path.
  - **Screen Shake**: Powerful visual feedback on every bounce.
  - **Particle Systems**: Explosive neon effects on paddle and wall hits.
- **Custom Pixel Font**: A custom-built geometry font that works even if your system's font modules are broken.
- **Auto-Scaling Fullscreen**: Automatically scales to any monitor resolution (1080p, 4K, etc.) while maintaining aspect ratio.
- **Competitive Rules**: First to 3 points wins, with a 3-2-1 countdown after every score.

## 🚀 Installation

1. **Clone the repository**:
   ```bash
   git clone <repo-url>
   cd ping-pong-cv
   ```

2. **Create a virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 🎮 How to Play

### Running the Game
For standard performance:
```bash
python main.py
```

**For Maximum Performance (RTX GPUs on Linux)**:
If you have an NVIDIA GPU (like an RTX 2050), use this command to ensure the AI runs on your dedicated card:
```bash
__NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia python main.py
```

### Controls
| Action | Key / Gesture |
| :--- | :--- |
| **Start Match** | `SPACE` |
| **Left Player** | Index Finger (Camera) or `W` / `S` |
| **Right Player** | Index Finger (Camera) or `UP` / `DOWN` |
| **Toggle Fullscreen** | `F` |
| **Exit Game** | `ESC` |

## 🛠️ Technology Stack
- **Vision**: [MediaPipe Tasks API](https://developers.google.com/mediapipe/solutions/vision/hand_landmarker) (Asynchronous Live Stream Mode)
- **Graphics**: [Pygame](https://www.pygame.org/) (SCALED & DOUBLEBUF)
- **Language**: Python 3.10+
- **Inference**: TFLite with GPU Delegate

---
Developed for **GDG ZU**. Enjoy the match! 🏓⚡️🏁
