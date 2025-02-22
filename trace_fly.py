import cv2
import numpy as np

# Initialize the video capture
#cap = cv2.VideoCapture('1637.MOV')
#cap = cv2.VideoCapture('videos/1116-c.MOV') #moving
#cap = cv2.VideoCapture('videos/1122-a.MOV') #moving
#cap = cv2.VideoCapture('videos/1117-a.MOV') #moving
#cap = cv2.VideoCapture('videos/144635-a-moving.mov') #moving
#cap = cv2.VideoCapture('videos/144830-a-moving-gromming.mov') #moving
#cap = cv2.VideoCapture('videos/145423-a-moving.mov') #moving
#cap = cv2.VideoCapture('videos/144408-a-moving.mov') #moving
#cap = cv2.VideoCapture('videos/6103-sick.mov')
#cap = cv2.VideoCapture('videos/6104-sick.mov')
#cap = cv2.VideoCapture('videos/6105-sick.mov')
#cap = cv2.VideoCapture('videos/6105-sick-1.mov')
#cap = cv2.VideoCapture('videos/6222_aloe_sick.mov')
#cap = cv2.VideoCapture('videos/6221_aloe_sick.mov')
#cap = cv2.VideoCapture('videos/6219_aloe_sick.mov')
#cap = cv2.VideoCapture('videos/6106-sick.mov')
#cap = cv2.VideoCapture('videos/6115_c.mov')
#cap = cv2.VideoCapture('videos/6116_c.mov')
#cap = cv2.VideoCapture('videos/6117_c.mov')
#cap = cv2.VideoCapture('videos/6118_c.mov')
#cap = cv2.VideoCapture('videos/6119_c.mov')
#cap = cv2.VideoCapture('videos/6120_c.mov')
#cap = cv2.VideoCapture('videos/6087-c-nd-irra.MOV') # good
#cap = cv2.VideoCapture('videos/1128-a.MOV') 
#cap = cv2.VideoCapture('videos/1116-c.MOV')
#cap = cv2.VideoCapture('videos/1638-sick.MOV')
#cap = cv2.VideoCapture('IMG_5508.MOV')
#cap = cv2.VideoCapture('videos/5488_old.MOV')
#cap = cv2.VideoCapture('videos/petri-c-002504.mov') #m
#cap = cv2.VideoCapture('videos/164313.mp4') #m

#age
#cap = cv2.VideoCapture('videos/63953_age_p.mp4') #P
#cap = cv2.VideoCapture('videos/63700.mp4') #P
#cap = cv2.VideoCapture('videos/154921_age.mp4') #
#cap = cv2.VideoCapture('videos/155017.mp4') #
#cap = cv2.VideoCapture('videos/155114.mp4') 

#cap = cv2.VideoCapture('videos/age-c60/195018-c60.mp4') 
#cap = cv2.VideoCapture('videos/age-c60/195056-c60.mp4') 
#cap = cv2.VideoCapture('videos/age-c60/195224-c60.mp4') 

#cap = cv2.VideoCapture('videos/age-petri-60/c60-petri.mov') 
#cap = cv2.VideoCapture('videos/age-petri-60/aloe-petri.mov') 
cap = cv2.VideoCapture('videos/age-petri/petri-sick.mov') 

#control
#cap = cv2.VideoCapture('videos/203102_c_p.mp4') #P
#cap = cv2.VideoCapture('videos/203204.mp4') #P
#cap = cv2.VideoCapture('videos/202833.mp4') #P
#cap = cv2.VideoCapture('videos/202918.mp4') #P
#cap = cv2.VideoCapture('videos/203014.mp4') #P

#cap = cv2.VideoCapture('videos/204757_c.mp4') 
#cap = cv2.VideoCapture('videos/205006.mp4') 
#cap = cv2.VideoCapture('videos/205054.mp4') 
#cap = cv2.VideoCapture('videos/205144.mp4') 


backSub = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
# stores the positions of fly in the format {id: [(x1, y1), (x2, y2), ...], ...}
fly_tracks = {}
fly_id_counter = 0
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fgMask = backSub.apply(gray)
    contours, _ = cv2.findContours(fgMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # temporary storage for current frame's detections
    current_detections = []

    for contour in contours:
        if cv2.contourArea(contour) > 100:  # Filter out small contours
        #if cv2.contourArea(contour) > 50:  # Filter out small contours    
            x, y, w, h = cv2.boundingRect(contour)
            center = (int(x+w/2), int(y+h/2))
            current_detections.append(center)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)  # mark center

    # track or update fly positions
    for detection in current_detections:
        # find the closest track (if any) to this detection
        closest_id = None
        min_distance = float('inf')
        for fly_id, positions in fly_tracks.items():
            last_position = positions[-1]
            distance = np.linalg.norm(np.array(last_position) - np.array(detection))
            if distance < min_distance:
                min_distance = distance
                closest_id = fly_id
        
        # if a track is close enough, update it; otherwise, start a new track
        if closest_id is not None and min_distance < 50:  # Threshold for "close enough"
        #PY!if closest_id is not None and min_distance < 15:  # Threshold for "close enough"    
            fly_tracks[closest_id].append(detection)
        else:
            fly_tracks[fly_id_counter] = [detection]
            fly_id_counter += 1

    #visualize tracks
    for fly_id, positions in fly_tracks.items():
        # print("fly_id:", fly_id)
        for i in range(1, len(positions)):
            #cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 2) #blue 
            #cv2.line(frame, positions[i - 1], positions[i], (255, 0, 0), 8) #blue 
            #cv2.line(frame, positions[i - 1], positions[i], (0, 255, 0), 2) #green 
            #cv2.line(frame, positions[i - 1], positions[i], (0, 100, 0), 2) #dark green
            cv2.line(frame, positions[i - 1], positions[i], (0, 255, 255), 2) #yellow 
            #cv2.line(frame, positions[i - 1], positions[i], (0, 255, 255), 8)
            #cv2.line(frame, positions[i - 1], positions[i], (230, 216, 173), 2) #skyblue 
            #cv2.line(frame, positions[i - 1], positions[i], (80, 0, 80), 2) #purple 
            #cv2.line(frame, positions[i - 1], positions[i], (128, 128, 128), 2) #light grey 
            #cv2.line(frame, positions[i - 1], positions[i], (139, 0, 0), 2) #dark blue 

    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#yellow (255, 255, 0)
#red (255, 0, 0)
#green  (0, 255, 0)  C60
#blue (0, 0, 255)
#sky blue (173, 216, 230)
#light grey (128, 128, 128)
