from ultralytics import YOLO
from tqdm import tqdm
import supervision as sv

PLAYER_DETECTION_MODEL = YOLO("foatball/weights/foatball350.pt")  # put your trained weights here



def extrack_player_crops(source,stride):
    frame_generator = sv.get_video_frames_generator(source,stride=stride)
    crops = []
    for frame in tqdm(frame_generator):  # process only first 100 frames for demo
        results = PLAYER_DETECTION_MODEL.track(frame, persist=True)[0]
        detections = sv.Detections.from_ultralytics(results)
        detections = detections.with_nms(threshold=0.3,class_agnostic=True)
        detections = detections[ detections.class_id == 2 ]  # keep only class_id 0 (players)
        crops += [
            sv.crop_image(frame,xyxy)
            for xyxy in detections.xyxy
            ]
        print(f"Cropped {len(crops)} player images from frame.")
    return crops