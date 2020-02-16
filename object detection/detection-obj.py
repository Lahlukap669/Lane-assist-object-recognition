import cv2
import numpy as np
import matplotlib.pylab as plt
from scipy.ndimage import rotate

vidcap = cv2.VideoCapture('input1.avi')
success,image = vidcap.read()
count = 0
alpha = 2 # Contrast control (1.0-3.0)
beta = -140 # Brightness control (0-100)
img_array = []
net = cv2.dnn.readNet("yolov3-tiny.weights", "yolov3-tiny.cfg")
razredi = []
razredi1 = []

with open("coco.names", "r") as datoteka:
    razredi = [line.strip() for line in datoteka.readlines()]
with open("coco1.names", "r") as datoteka:
    razredi1 = [line.strip() for line in datoteka.readlines()]
      
imena_plasti = net.getLayerNames()
izhodne_plasti = [imena_plasti[i[0] - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 190, size=(len(razredi), 3))

while success:
##    image =  rotate(image, -90)
##    image = image[500:1000, 0:720]

    #image = cv2.imread("result0.png")
    visina, sirina, kanali = image.shape


    #izluščevanje podrobnosti iz slike
    blob = cv2.dnn.blobFromImage(image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(izhodne_plasti)

    # Showing informations on the screen
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.4:
                # Object detected
                center_x = int(detection[0] * sirina)
                center_y = int(detection[1] * visina)
                w = int(detection[2] * sirina)
                h = int(detection[3] * visina)
                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)


    font = cv2.FONT_HERSHEY_SIMPLEX
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(razredi1[class_ids[i]])
            color = colors[i]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            cv2.putText(image, label, (x, y + 30), font, 0.5, color, 1)

    slika = "result/result"+str(count)+".png"
    plt.imshow(image)
##    plt.show()
    img_array.append(image)
    success,image = vidcap.read()
    count += 1
    print(count)

out = cv2.VideoWriter('final1.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (720, 500))
 
for i in range(len(img_array)):
    out.write(img_array[i])
out.release()
