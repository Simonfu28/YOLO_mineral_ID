import cv2
import argparse
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True,
                help = 'path to test image')
args = ap.parse_args()

weights_path = '/home/simon/yolo_mineral_id/models/chalcopyrite1.weights'
config_path = '/home/simon/yolo_mineral_id/models/chalcopyrite1.cfg' 

classes_path = '/home/simon/yolo_mineral_id/models/chalcopyrite1_classes.txt'
classes = None
with open(classes_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
##COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
COLORS = [[245, 240, 240], [0, 223, 255]]

image = cv2.imread(args.image) ## read test image
Width = image.shape[1]
Height = image.shape[0]
scale = 0.00392

net = cv2.dnn.readNet(weights_path, config_path)
blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)


def get_output_layers(net):
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):
    mineral = str(classes[class_id])
    conf = str(confidence)
    conf = "c: " + conf[:5]
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, mineral, (x-10,y-10), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)
    cv2.putText(img, conf, (x-10, y_plus_h+15), cv2.FONT_HERSHEY_DUPLEX, 0.5, color, 1)

outs = net.forward(get_output_layers(net))

class_ids = []
confidences = []
boxes = []
conf_thres = 0.5
nms_thres = 0.4

for out in outs:
    for detection in out:
        scores = detection[5:]
        class_id = np.argmax(scores)
        confidence = scores[class_id]
        if confidence > 0.5:
            center_x = int(detection[0] * Width)
            center_y = int(detection[1] * Height)
            w = int(detection[2] * Width)
            h = int(detection[3] * Height)
            x = center_x - w/2
            y = center_y - h/2
            class_ids.append(class_id)
            confidences.append(float(confidence))
            boxes.append([x, y, w, h])

indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, nms_thres)

for i, n in enumerate(indices):
    box = boxes[i]
    x = box[0]
    y = box[1]
    w = box[2]
    h = box[3]

    draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))



cv2.imshow("chalcopyrite1_test", image)
cv2.waitKey()
cv2.imwrite("chalcopyrite1_test.jpg", image)
cv2.destroyAllWindows()
