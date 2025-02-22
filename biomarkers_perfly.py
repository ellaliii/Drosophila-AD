import cv2
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

# extract features for each fly
def extract_features_per_fly(video_path, output_csv):
    cap = cv2.VideoCapture(video_path)
    backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    
    # store all positions with frame info
    positions = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        fg_mask = backSub.apply(gray)

        # find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) > 100:  # filter noise
                x, y, w, h = cv2.boundingRect(contour)
                cx, cy = x + w // 2, y + h // 2
                positions.append([frame_count, cx, cy])

        frame_count += 1

    cap.release()

    # clustering flies
    positions = np.array(positions)
    clustering = DBSCAN(eps=20, min_samples=5).fit(positions[:, 1:])
    positions = np.hstack((positions, clustering.labels_.reshape(-1, 1)))

    # filter noise (label -1 means noise in DBSCAN)
    positions = positions[positions[:, -1] != -1]
    positions_df = pd.DataFrame(positions, columns=["frame", "x", "y", "fly_id"])

    # calculate features per fly
    features = []
    for fly_id, group in positions_df.groupby("fly_id"):
        group = group.sort_values("frame")
        deltas = np.diff(group[["x", "y"]].values, axis=0)
        speeds = np.sqrt((deltas**2).sum(axis=1))
        directions = np.arctan2(deltas[:, 1], deltas[:, 0])
        frame_intervals = np.diff(group["frame"].values)

        features.append({
            "fly_id": int(fly_id),
            "avg_speed": speeds.mean(),
            "max_speed": speeds.max(),
            "direction_changes": (np.abs(np.diff(directions)) > 0.1).sum(),
            "movement_frequency": len(group) / frame_count,
        })

    # save features to CSV
    features_df = pd.DataFrame(features)
    features_df.to_csv(output_csv, index=False)
    print(f"Features saved to {output_csv}")

    return features_df

# main function
if __name__ == "__main__":
    name="6105-sick"
    video_path = "../videos/"+name+".mov"  # path to the video file
    output_csv = "fly_features_"+name+".csv"  # path to save the extracted features

    features_df = extract_features_per_fly(video_path, output_csv)
