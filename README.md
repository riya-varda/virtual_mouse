<h1 align="center">🖐️ Virtual Mouse</h1>
<h3 align="center">Gesture Control System v2.0</h3>
<p align="center">
  Control your computer using just your hand ✨  
  <br>
  Built with Computer Vision + MediaPipe
</p>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv">
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Tracking-orange?style=for-the-badge">
  <img src="https://img.shields.io/badge/Status-Active-success?style=for-the-badge">
</p>

---

## 🚀 About The Project

**Virtual Mouse** is a real-time gesture-based control system that replaces your physical mouse using hand tracking.

It uses your webcam to detect hand movements and converts them into actions like cursor movement, clicking, scrolling, and even system controls like volume and brightness.

No hardware required — just your camera.

---

## ✨ Features

- 🎯 Smooth cursor control using index finger
- 👆 Pinch gestures for click & drag
- 🖱️ Left click, right click, double click
- 🔄 Scroll using two fingers
- 🎵 Media control (Play/Pause, Next, Previous)
- 🔊 Volume control & mute
- 💡 Brightness control (Windows)
- 📸 Screenshot using fist gesture
- ⚡ Kalman Filter for smooth movement
- 🧠 Gesture State Machine to avoid false triggers
- 📊 Real-time UI with FPS and gesture feedback

---

## 🧠 Gesture Controls

<table align="center">
<tr><th>Gesture</th><th>Action</th></tr>
<tr><td>☝️ Index finger</td><td>Move cursor</td></tr>
<tr><td>🤏 Pinch</td><td>Left click</td></tr>
<tr><td>🤏🤏 Double pinch</td><td>Double click</td></tr>
<tr><td>🤏 + move</td><td>Drag</td></tr>
<tr><td>✌️ + pinch</td><td>Right click</td></tr>
<tr><td>✌️ vertical</td><td>Scroll</td></tr>
<tr><td>🖖 swipe right</td><td>Next track</td></tr>
<tr><td>🖖 swipe left</td><td>Previous track</td></tr>
<tr><td>🖐️ Open palm</td><td>Play / Pause</td></tr>
<tr><td>✊ Fist (hold)</td><td>Screenshot</td></tr>
<tr><td>👍 Thumb up</td><td>Volume up</td></tr>
<tr><td>🤙 Pinky up</td><td>Volume down</td></tr>
<tr><td>🤘 Rock sign</td><td>Mute</td></tr>
<tr><td>☝️ Ring finger</td><td>Brightness up</td></tr>
<tr><td>🤙 Ring + Pinky</td><td>Brightness down</td></tr>
</table>

---

## 🛠️ Tech Stack

- **Python**
- **OpenCV**
- **MediaPipe (Tasks API)**
- **PyAutoGUI**
- **NumPy**

---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/virtual_mouse.git
cd virtual_mouse
pip install -r requirements.txt
```

## ▶️ Run the Project

```bash
python main.py
```

> Press `ESC` to exit

---

## 📦 Model Download

The required model will automatically download on first run:

- `hand_landmarker.task` (~9MB)

---

## 🧩 How It Works

1. 📷 Captures webcam input
2. ✋ Detects hand landmarks using MediaPipe
3. 🧠 Processes gestures using a state machine
4. 🎯 Maps gestures to system controls
5. ⚡ Smoothens motion using Kalman filtering

---

## ⚠️ Requirements

- Python 3.8+
- Webcam
- Good lighting for accurate detection

---

## 💡 Future Improvements

- Multi-hand support
- Custom gesture mapping
- GUI for configuration
- Cross-platform brightness control
- Performance optimization

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork this repo and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

<p align="center">Made with ❤️ using Computer Vision</p>
