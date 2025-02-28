import cv2
import numpy as np

# initialize the video capture
cap = cv2.VideoCapture('52.mp4')
#cap = cv2.VideoCapture('ControlStable.mp4')

if not cap.isOpened():
    print("Error opening video stream or file")

# get the frames per sec of the video
fps = cap.get(cv2.CAP_PROP_FPS)

backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

# initialize tracking information
planaria_tracks = {}
planaria_id_counter = 0
# speed calculation (in pixels/sec)
planaria_speeds = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgMask = backSub.apply(gray)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    current_detections = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # filter out small contours
            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x+w/2), int(y+h/2))
            current_detections.append(center)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # mark center

    for detection in current_detections:
        closest_id = None
        min_distance = float('inf')
        for planaria_id, positions in planaria_tracks.items():
            last_position = positions[-1]
            distance = np.linalg.norm(np.array(last_position) - np.array(detection))
            if distance < min_distance:
                min_distance = distance
                closest_id = planaria_id

        if closest_id is not None and min_distance < 50:  # threshold for "close enough"
            planaria_tracks[closest_id].append(detection)
            # calculate speed
            if len(planaria_tracks[closest_id]) > 1:
                # speed = distance / time
                # distance is the Euclidean distance between the last two positions
                # time is 1 frame duration (1/fps)
                speed = min_distance * fps  
                planaria_speeds[closest_id] = speed
        else:
            planaria_tracks[planaria_id_counter] = [detection]
            planaria_speeds[planaria_id_counter] = 0  # initial speed is 0
            planaria_id_counter += 1

    for planaria_id, positions in planaria_tracks.items():
        if len(positions) > 1:
            cv2.putText(frame, f"ID: {planaria_id} Speed: {planaria_speeds[planaria_id]:.2f}px/s", positions[-1], cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            print(f"ID: {planaria_id} - Speed: {planaria_speeds[planaria_id]:.2f}px/s")
        for i in range(1, len(positions)):
            #cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 2)
            cv2.line(frame, positions[i - 1], positions[i], (0, 255, 255), 2) #yellow 

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
