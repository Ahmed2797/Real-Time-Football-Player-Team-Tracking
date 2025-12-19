# Real-Time Football Player & Team Tracking

![Football Tracking](assets/(image-1.png))  <!-- Optional: add a screenshot -->

![alt text](image-2.png)

This project demonstrates **real-time football player detection, team classification, and event annotation** using **YOLO, Sports library, and Supervision**.

---

## Features

- **Player, Goalkeeper, Referee & Ball Detection**  
  Detect all key entities on the pitch in real-time.

- **Team Classification**  
  Classify players into teams using `sports.common.team.TeamClassifier`.

- **Tracking**  
  Maintain unique IDs for players using `ByteTrack`.

- **Frame Annotation**  
  Draw bounding boxes, labels, ellipses, and triangles for visual clarity.

- **Video Output**  
  Generates an annotated video highlighting players, teams, ball, and referees.

---

## Tech Stack

- **Computer Vision:** OpenCV, Supervision  
- **Object Detection:** YOLO (Ultralytics)  
- **Tracking:** ByteTrack  
- **Team Classification:** Roboflow Sports Library  
- **Deep Learning Framework:** PyTorch  
- **Environment:** Python 3.10, Conda  

---

## Installation

```bash
# Clone repo
git clone https://github.com/Ahmed2797/Real-Time-Football-Player-Team-Tracking.git
cd Foatball_Player

# Create environment
conda create -n football python=3.10 -y
conda activate football

# Install dependencies
pip install -r requirements.txt


echo "# Real-Time-Football-Player-Team-Tracking" >> README.md
git init
git add .
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/Ahmed2797/Real-Time-Football-Player-Team-Tracking.git
git push -u origin main
