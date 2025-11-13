HIWI — Panel Status Monitoring + Servo Control

This repository contains two coordinated components:

Component	Runs on	Purpose
viewer_gradmap_judge.py	PC / Laptop	Performs panel status classification and Grad-CAM visualization.
servo_gamepad_tamed.py	Raspberry Pi	Controls the camera orientation via a USB gamepad.


⸻

Repository Structure

HIWI/
├─ viewer_gradmap_judge.py       # PC-side camera viewer + Grad-CAM + status inference
├─ model_def.py                  # Model definition for loading saved checkpoint
├─ panel_cls_full.pt             # Trained classifier model with metadata
├─ servo_gamepad_tamed.py        # Raspberry Pi dual-servo control via USB gamepad
└─ README.md


⸻

1) Hardware Setup

PC Side (Visual Recognition)
	•	Runs viewer_gradmap_judge.py
	•	Requires a USB camera

Raspberry Pi Side (Servo Control)
	•	Raspberry Pi (any version with GPIO)
	•	Two SG90 servos:
	•	Horizontal (yaw): GPIO17
	•	Vertical (pitch): GPIO27
	•	External 5V servo power recommended (≥1A)
	•	Important: Connect servo GND and Pi GND together (common ground)

⸻

2) Software Setup

PC Environment

Install dependencies:

pip install opencv-python numpy pillow matplotlib
pip install torch torchvision

Run:

python3 viewer_gradmap_judge.py --overlay_cam

Controls in viewer:

Action	Key
Draw ROI	Click + drag
Toggle Grad-CAM overlay	O
Reset ROI	R
Quit	Q


⸻

Raspberry Pi Environment

Install required packages:

sudo apt update
sudo apt install -y python3-pip python3-opencv pigpio
sudo systemctl enable --now pigpiod

Run servo controller:

export SDL_VIDEODRIVER=dummy
python3 servo_gamepad_tamed.py

Gamepad inputs:

Control	Function
Left stick (left/right)	Camera yaw
Left stick (up/down)	Camera pitch
A / Start	Re-center servos
Ctrl + C	Exit

Movement is smooth and slow, but reactions are instant to joystick changes.

⸻

3) How to Use Both Together (Typical Operation)
	1.	On PC, run:

python3 viewer_gradmap_judge.py --overlay_cam


	2.	On Raspberry Pi, run:

python3 servo_gamepad_tamed.py


	3.	Use the gamepad to point the camera at the indicator lights
→ The PC program will classify LED status and show Grad-CAM reasoning heatmaps.

⸻

4) Important Note

The classification model runs on the PC, not on the Raspberry Pi,
because Grad-CAM and CNN inference require higher compute performance.

The Raspberry Pi is used only for servo control.

⸻
