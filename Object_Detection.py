import cv2
import numpy as np
import math
import time
net = cv2.dnn.readNet('yolov3_training_final.weights', 'yolov3_testing.cfg')

classes = ['User']
#with open("classes.txt", "r") as f:
 #   classes = f.read().splitlines()

#cap = cv2.VideoCapture('video4.mp4')
#cap = 'test_images/<your_test_image>.jpg'
font = cv2.FONT_HERSHEY_PLAIN
colors = np.random.uniform(0, 255, size=(100, 3))

while True:
    start_time = time.perf_counter()
    capture = cv2.VideoCapture(0)
    _, img = capture.read()
    #img = cv2.imread("boo1.jpg")
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []
    angle = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.2, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            x0= int(width/2)
            y0= height
            central_x = int(x+w/2)
            central_y = int(y+h/2)
            color = colors[1]
            color_1 = color[2]
            direction = int(math.atan2(central_y - y0, center_x - x0)*-1*180/math.pi)
            angle.append(direction)
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.putText(img, label + " " + confidence, (x, y+20), font, 1, (255,255,255), 2)
            cv2.line(img, (x0,y0), (int(central_x),int(central_y)), color_1, 3)

    for idx,angle in enumerate(angle):
        cv2.putText(img, "angle: "+ str(angle) , (10, 10+20*idx), font, 1, (0,255,0), 2)
    difference = time.perf_counter() - start_time
    string = "fps= " + str(round(1/difference,2))
    cv2.putText(img, string , (width-150, 30), font, 1, (255,255,255), 2)
    cv2.imshow('Image', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()