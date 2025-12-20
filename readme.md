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

## Model

    https://drive.google.com/file/d/1gIQuv32iJtyvfoxLBkG6T2Fklq2P2TBz/view?usp=sharing

## video_dowload

    !gdown -O "0bfacc_0.mp4" "https://drive.google.com/uc?id=12TqauVZ9tLAv8kWxTTBFWtgt2hNQ4_ZF"
    !gdown -O "2e57b9_0.mp4" "https://drive.google.com/uc?id=19PGw55V8aA6GZu5-Aac5_9mCy3fNxmEf"
    !gdown -O "08fd33_0.mp4" "https://drive.google.com/uc?id=1OG8K6wqUw9t7lp9ms1M48DxRhwTYciK-"
    !gdown -O "573e61_0.mp4" "https://drive.google.com/uc?id=1yYPKuXbHsCxqjA9G-S6aeR2Kcnos8RPU"
    !gdown -O "121364_0.mp4" "https://drive.google.com/uc?id=1vVwjW1dE1drIdd4ZSILfbCGPD4weoNiu"

## Installation

    git clone https://github.com/Ahmed2797/Real-Time-Football-Player-Team-Tracking.git
    conda create -n football python=3.10 -y
    conda activate football
    pip install -r requirements.txt

## Final notebook here

    https://drive.google.com/file/d/1d33mYdw9VX7agOOGx_PgHp6ZVZykhzdv/view?usp=sharing
