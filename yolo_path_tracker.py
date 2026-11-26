import cv2
import numpy as np
from tracker import Tracker

PROBABILITY_MINIMUM = 0.5
THRESHOLD = 0.3

def post_processing(frame,net_output) :
    bounding_boxes = []
    confidences = []
    h, w = frame.shape[:2]
    for result in net_output:
        for detection in result:
            scores = detection[5:]
            class_current = np.argmax(scores)
            confidence_current = scores[class_current]
            #Keeps only people
            if confidence_current > PROBABILITY_MINIMUM and class_current == 0:
                box_current = detection[0:4] * np.array([w, h, w, h])
                x_center, y_center, box_width, box_height = box_current.astype('int')
                x_min = int(x_center - (box_width / 2))
                y_min = int(y_center - (box_height / 2))
                bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
                confidences.append(float(confidence_current))
    return bounding_boxes,confidences

def update_tracker(tracker,bounding_boxes,nms_results):
    centroids = []
    #This variable below stores the indices of the bounding boxes that we want to keep
    #from the bounding box list
    kept_boxes = {}
    if len(nms_results) > 0:
        for i in nms_results.flatten():
            x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
            box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]
            centroid =  (x_min + box_width/2 , y_min + box_height/2)
            centroids.append(centroid)
            kept_boxes[centroid] = i

    objects = tracker.update(centroids)
    paths = tracker.get_paths()

    return kept_boxes, objects, paths

def draw_on_frame(frame,tracker,objects,paths,bounding_boxes,kept_boxes):
    #Note, the random color generation doesn't work like it does in A1, since only one Class
    coco_labels = 80
    np.random.seed(42)
    colours = np.random.randint(0, 255, size=(coco_labels, 3), dtype='uint8')

    for id,centroid in objects.items():
        if tracker.lost_frames[id] > 0:
            #Means that the object is not seen in this frame
            continue
        i = kept_boxes[centroid]
        x_min, y_min, box_width, box_height = bounding_boxes[i]
        cx, cy = int(centroid[0]),int(centroid[1])
        colour_box = [int(j) for j in colours[id % coco_labels]]

        #Bounding Box plus filled circle in the middle
        cv2.rectangle(frame, (x_min, y_min), (x_min + box_width, y_min + box_height),
                                colour_box, 3)
        cv2.circle(frame, (cx,cy), 10, colour_box, -1)
        text = f"{id}"
        cv2.putText(frame, text, (cx, cy - 7), cv2.FONT_HERSHEY_SIMPLEX, 1.5, colour_box, 5)

        #Trail behind the circle
        #Only draw the latest 15 points
        centroid_points = np.array(paths[id][-15:],dtype=np.int32)
        cv2.polylines(frame,[centroid_points],False,colour_box, 2)


def main():
    # Load YOLO model
    net = cv2.dnn.readNetFromDarknet("yolo_files/yolov3.cfg","yolo_files/yolov3.weights")
    yolo_layers = ['yolo_82', 'yolo_94', 'yolo_106']

    # Load video
    video_path = 'input.webm'
    cap = cv2.VideoCapture(video_path)

    # Initialize the tracker
    tracker = Tracker()

    # Determine the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Set up the video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # For MP4 format
    out_vid = cv2.VideoWriter('output.mp4', fourcc, 15.0, (int(cap.get(3)), int(cap.get(4))))

    # Process each frame
    for _ in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        input_blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        # - Use YOLO net to detect objects
        net.setInput(input_blob)
        output = net.forward(yolo_layers)

        bounding_boxes, confidences = post_processing(frame,output)
                
        # - Apply Non-Maximum Suppression (NMS) to filter out overlapping boxes
        results = cv2.dnn.NMSBoxes(bounding_boxes, confidences, PROBABILITY_MINIMUM, THRESHOLD)

        # - Update the tracker with the centroids of detected objects
        kept_boxes, objects, paths = update_tracker(tracker,bounding_boxes,results)

        # - Draw bounding boxes, trails, and labels on the frame
        draw_on_frame(frame,tracker,objects,paths,bounding_boxes,kept_boxes)

        # Write the processed frame to the output video
        out_vid.write(frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video writer and video capture objects
    out_vid.release()
    cap.release()
    cv2.destroyAllWindows()

    return 0

if __name__ == '__main__': #runs main when file is executed directly.
    raise SystemExit(main())
