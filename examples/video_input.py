from facetorch import FaceAnalyzer
from omegaconf import OmegaConf
import torchvision
import cv2


path_config="notebooks/gpu.config.yml"
cfg = OmegaConf.load(path_config)
analyzer = FaceAnalyzer(cfg.analyzer)
video_path = "notebooks/output.mp4"
video = cv2.VideoCapture(video_path)

preds = {}
if not video.isOpened():
    print("Error opening video file")
    exit()

fps = video.get(cv2.CAP_PROP_FPS)

# Set interval to capture frames (every second)
frame_interval = int(fps)  # Change this value to alter frame extraction frequency
frame_count = 0

while True:
    ret, frame = video.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    if not ret:
        break

    # Extract frame at every 'frame_interval'
    if frame_count % frame_interval == 0:
        print(frame_count)
        response = analyzer.run(
                image=frame,
                batch_size=8,
                include_tensors=True,
                return_img_data=True,
                fix_img_size = True
            )
       

    frame_count += 1
    if frame_count == 1000:
        print('done')
        break
    
video.release()
