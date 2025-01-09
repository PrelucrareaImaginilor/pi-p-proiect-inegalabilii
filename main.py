import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#incarc modelul YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

#incarc clipul
video_input_file_name = "volley3.mp4"
video = cv2.VideoCapture(video_input_file_name)
ok, frame = video.read()

if not ok:
    print("Could not open video")
    exit()

#dimensiuni clip
height, width, _ = frame.shape

#detecteaza jucatorii in primul cadru
blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

#lista pt bboxes detectate
bboxes = []
confidences = []
class_ids = []

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5 and classes[class_id] == "person":
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width * 1.5)  # Increase box width by 50%
            h = int(detection[3] * height)
            x = int(center_x - w / 2)
            y = int(center_y - h / 2)
            bboxes.append((x, y, w, h))
            confidences.append(float(confidence))
            class_ids.append(class_id)

#elimina duplicatele folosind Non-Maximum Suppression (NMS)
indices = cv2.dnn.NMSBoxes(bboxes, confidences, 0.5, 0.4)

if len(indices) > 0:
    selected_bboxes = [bboxes[i] for i in indices.flatten()]
else:
    selected_bboxes = []


tracked_bboxes = []

#redimensionare a clipului
resize_width, resize_height = 1200, 600
scale_x = width / resize_width
scale_y = height / resize_height

def click_and_select(event, x, y, flags, param):
    global tracked_bboxes
    if event == cv2.EVENT_LBUTTONDOWN:
        #conversia coordonatele redimensionate la cele original
        original_x = int(x * scale_x)
        original_y = int(y * scale_y)
        for bbox in selected_bboxes:
            x_min, y_min, w, h = bbox
            if x_min < original_x < x_min + w and y_min < original_y < y_min + h:
                tracked_bboxes.append(bbox)
                break

cv2.namedWindow("Select Players")
cv2.setMouseCallback("Select Players", click_and_select)

while True:
    resized_frame = cv2.resize(frame.copy(), (resize_width, resize_height))
    cv2.imshow("Select Players", resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyWindow("Select Players")

#inițializeaza trackerele pe baza bboxes selectate
trackers = []
for bbox in tracked_bboxes:
    tracker = cv2.TrackerKCF_create()
    tracker.init(frame, tuple(bbox))
    trackers.append(tracker)

#matrice pentru heatmap
heatmap_data = np.zeros((height, width))

video_output_file_name = "volley-output.mp4"
video_out = cv2.VideoWriter(video_output_file_name, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))

#loop pentru procesarea video
while True:
    ok, frame = video.read()
    if not ok:
        break

    for tracker in trackers:
        ok, bbox = tracker.update(frame)
        if ok:
            x, y, w, h = [int(v) for v in bbox]
            center_x = x + w // 2
            center_y = y + h // 2

            #adauga coordonatele bboxului in heatmap
            for i in range(-5, 6):  # Y axis range
                for j in range(-5, 6):  # X axis range
                    if 0 <= int(center_y + i) < height and 0 <= int(center_x + j) < width:
                        heatmap_data[int(center_y + i), int(center_x + j)] += 10

            #deseneaza dreptunghiul pe jucător
            p1 = (x, y)
            p2 = (x + w, y + h)
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)

    resized_frame = cv2.resize(frame, (resize_width, resize_height))
    cv2.imshow("Tracking", resized_frame)
    video_out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
video_out.release()
cv2.destroyAllWindows()

#normalizeaza heatmap-ul pentru vizualizare
heatmap_data = heatmap_data / heatmap_data.max()

#afișeaza heatmap-ul folosind seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', cbar=True, xticklabels=False, yticklabels=False)
plt.title("Combined Heatmap - Player Movement")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()