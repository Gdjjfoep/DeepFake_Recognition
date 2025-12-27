import cv2
import numpy as np

def crop_center_square(frame):
    y, x = frame.shape[0:2]
    min_dim = min(y, x)
    start_x = (x // 2) - (min_dim // 2)
    start_y = (y // 2) - (min_dim // 2)
    return frame[start_y : start_y + min_dim, start_x : start_x + min_dim]

def load_video(path, max_frames=20, resize=(224, 224)):
    cap = cv2.VideoCapture(path)
    frames = []
    try:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0: return None
        skip_frames_window = max(int(total_frames / max_frames), 1)

        for i in range(max_frames):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i * skip_frames_window)
            ret, frame = cap.read()
            if not ret: break
            frame = crop_center_square(frame)
            frame = cv2.resize(frame, resize)
            frame = frame[:, :, [2, 1, 0]] # BGR to RGB
            frames.append(frame)
    finally:
        cap.release()
        
    frames = np.array(frames)
    if len(frames) < max_frames:
        if len(frames) == 0: return None
        padding = np.zeros((max_frames - len(frames), *resize, 3), dtype="uint8")
        frames = np.concatenate((frames, padding), axis=0)
    return frames