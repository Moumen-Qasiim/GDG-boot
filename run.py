#!/usr/bin/env python3
import os
import sys
import subprocess
import venv
from pathlib import Path

# --- Configuration ---
PROJECT_ROOT = Path(__file__).parent.resolve()
VENV_DIR = PROJECT_ROOT / ".venv"
GAMES = [
    {
        "name": "Rock Paper Scissors",
        "path": "Rock-paper-scissors/rock/rock/rock.py",
        "cwd": "Rock-paper-scissors/rock/rock"
    },
    {
        "name": "Snake",
        "path": "snake/snake.py",
        "cwd": "snake"
    },
    {
        "name": "Ping Pong CV",
        "path": "ping-pong-cv/main.py",
        "cwd": "ping-pong-cv"
    },
    {
        "name": "Tic Tac Toe CV",
        "path": "tic-tac-toe-cv/main.py",
        "cwd": "tic-tac-toe-cv"
    }
]

# --- Colors for UI ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"{Colors.HEADER}{Colors.BOLD}=== {text} ==={Colors.ENDC}")

def print_step(text):
    print(f"{Colors.CYAN}[*] {text}{Colors.ENDC}")

def print_success(text):
    print(f"{Colors.GREEN}[+] {text}{Colors.ENDC}")

def print_error(text):
    print(f"{Colors.FAIL}[!] {text}{Colors.ENDC}")

# --- Logic ---

def setup_venv():
    if not VENV_DIR.exists():
        print_step("Creating virtual environment...")
        venv.create(VENV_DIR, with_pip=True)
        print_success("Virtual environment created.")
    else:
        print_step("Virtual environment already exists.")

def get_pip_path():
    if os.name == 'nt':
        return VENV_DIR / "Scripts" / "pip.exe"
    return VENV_DIR / "bin" / "pip"

def get_python_path():
    if os.name == 'nt':
        return VENV_DIR / "Scripts" / "python.exe"
    return VENV_DIR / "bin" / "python"

def install_requirements():
    pip_path = get_pip_path()
    all_requirements = set()
    
    # Collect all requirements from subdirectories
    for req_file in PROJECT_ROOT.glob("**/requirements.txt"):
        if ".venv" in str(req_file):
            continue
        print_step(f"Reading {req_file.relative_to(PROJECT_ROOT)}...")
        with open(req_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    all_requirements.add(line)
    
    if not all_requirements:
        print_warning("No requirements found.")
        return

    # Create a temporary requirements file to install all at once
    temp_reqs = PROJECT_ROOT / "all_requirements.txt"
    with open(temp_reqs, 'w') as f:
        f.write("\n".join(sorted(all_requirements)))
    
    print_step("Installing dependencies... (This might take a while)")
    try:
        subprocess.check_call([str(pip_path), "install", "-r", str(temp_reqs)])
        print_success("Dependencies installed successfully.")
    except subprocess.CalledProcessError:
        print_error("Failed to install dependencies.")
    finally:
        if temp_reqs.exists():
            os.remove(temp_reqs)

def run_game(game):
    python_path = get_python_path()
    game_path = PROJECT_ROOT / game["path"]
    game_cwd = PROJECT_ROOT / game["cwd"]
    
    if not game_path.exists():
        print_error(f"Game file not found: {game_path}")
        return

    print_header(f"Launching {game['name']}")
    print_step(f"Running: {game['path']} in {game['cwd']}")
    
    try:
        # We use subprocess.run so the terminal returns to the menu after game exit
        subprocess.run([str(python_path), str(game_path)], cwd=str(game_cwd))
    except KeyboardInterrupt:
        print_step("\nGame exited by user.")

def main_menu():
    while True:
        os.system('clear' if os.name == 'posix' else 'cls')
        print_header("GDG-BOOT UNIFIED GAME LAUNCHER")
        print(f"{Colors.BLUE}Choose a game to play:{Colors.ENDC}")
        for i, game in enumerate(GAMES):
            print(f"  {Colors.BOLD}{i+1}.{Colors.ENDC} {game['name']}")
        print(f"  {Colors.BOLD}Q.{Colors.ENDC} Quit")
        
        choice = input(f"\n{Colors.CYAN}Selection: {Colors.ENDC}").strip().upper()
        
        if choice == 'Q':
            print_step("Goodbye!")
            break
        
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(GAMES):
                run_game(GAMES[idx])
                input(f"\n{Colors.BLUE}Press Enter to return to menu...{Colors.ENDC}")
            else:
                print_error("Invalid selection.")
                time.sleep(1)
        except ValueError:
            print_error("Please enter a number or 'Q'.")
            import time
            time.sleep(1)

def check_models():
    print_step("Checking Mediapipe models...")
    model_path = PROJECT_ROOT / "hand_landmarker.task"
    model_url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
    if not model_path.exists():
        print_step("Downloading hand_landmarker.task... (Required for tracking)")
        import urllib.request
        try:
            urllib.request.urlretrieve(model_url, str(model_path))
            print_success("Model downloaded.")
        except Exception as e:
            print_error(f"Failed to download model: {e}")
    else:
        print_success("Hand tracking model found.")

if __name__ == "__main__":
    try:
        print_header("INITIALIZING SETUP")
        setup_venv()
        install_requirements()
        check_models()
        main_menu()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)
