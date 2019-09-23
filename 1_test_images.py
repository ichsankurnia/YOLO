#!/usr/bin/env python
# coding: utf-8

# In[30]:


import cv2
from darkflow.net.build import TFNet
import matplotlib.pyplot as plt

# define the model options and run

options = {
    'model': 'cfg/yolov1.cfg',
    'load': 'bin/yolov2.weights'
}
# options = {
#     'model': 'cfg/yolo.cfg',
#     'load': 'bin/yolov2.weights',
#     'threshold': 0.3
# }

tfnet = TFNet(options)


img = cv2.imread('test-images/living_room.jpg', cv2.IMREAD_COLOR)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# use YOLO to predict the image
result = tfnet.return_predict(img)

img.shape


# pull out some info from the results

tl = (result[0]['topleft']['x'], result[0]['topleft']['y'])
br = (result[0]['bottomright']['x'], result[0]['bottomright']['y'])
label = result[0]['label']


# add the box and label and display it
img = cv2.rectangle(img, tl, br, (0, 255, 0), 7)
img = cv2.putText(img, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
# plt.imshow(img)
# plt.show()

cv2.imshow("image", img)
cv2.waitKey(0)