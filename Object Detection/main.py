from cv2 import cv2
import numpy as np

with open("COCO_labels.txt","r") as f:
    class_names=f.read().split("\n")
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))


model = cv2.dnn.readNet(model='frozen_inference_graph.pb',config='ssd_mobilenet_v2_coco_2018_03_29.pbtxt',framework='TensorFlow')

image=cv2.imread("test.jpg")
image_h,image_w,_=image.shape
blob=cv2.dnn.blobFromImage(image=image,size=(300,300),mean=(104,117,123),swapRB=True)
model.setInput(blob)
output=model.forward()

for detection in output[0, 0, :, :]:
   confidence = detection[2]
   if confidence > .4:
       class_id = detection[1]
       class_name = class_names[int(class_id)-1]
       color = COLORS[int(class_id)]
       box_x = detection[3] * image_w
       box_y = detection[4] * image_h
       box_w = detection[5] * image_w
       box_h = detection[6] * image_h
       cv2.rectangle(image, (int(box_x), int(box_y)), (int(box_w), int(box_h)),color, thickness=2)
       cv2.putText(image, class_name, (int(box_x), int(box_y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 1,color,2)
 
cv2.imshow('image', image)
cv2.imwrite('image_result.jpg', image)
cv2.waitKey(0)
cv2.destroyAllWindows()