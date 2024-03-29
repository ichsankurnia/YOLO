import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/yolo.cfg',
    'load': 'bin/yolov2.weights',
    'threshold': 0.15,
}

tfnet = TFNet(option)

capture = cv2.VideoCapture(0) # 'test-videos/car_chase_01.mp4'
colors = [tuple(255 * np.random.rand(3)) for i in range(5)]


while 1:
    stime = time.time()
    _, frame = capture.read()

    results = tfnet.return_predict(frame)

    for color, result in zip(colors, results):
        tl = (result['topleft']['x'], result['topleft']['y'])
        br = (result['bottomright']['x'], result['bottomright']['y'])
        label = result['label']
        confidence = result['confidence']
        text = '{}: {:.0f}%'.format(label, confidence * 100)
        cv2.rectangle(frame, tl, br, color, 7)
        cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

    cv2.imshow('frame', frame)
    print('FPS {:.1f}'.format(1 / (time.time() - stime)))
    
    if cv2.waitKey(1) == 27:
        break

capture.release()
cv2.destroyAllWindows()

# import cv2
# from darkflow.net.build import TFNet
# import numpy as np
# import time

# options = {
#     'model': 'cfg/yolo.cfg',
#     'load': 'bin/yolov2.weights',
#     'threshold': 0.2,
#     # 'gpu': 1.0
# }

# tfnet = TFNet(options)
# colors = [tuple(255 * np.random.rand(3)) for _ in range(10)]

# capture = cv2.VideoCapture(0)
# # capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# # capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

# while 1:
#     stime = time.time()
#     ret, frame = capture.read()
#     # if ret:
#     results = tfnet.return_predict(frame)
#     for color, result in zip(colors, results):
#         tl = (result['topleft']['x'], result['topleft']['y'])
#         br = (result['bottomright']['x'], result['bottomright']['y'])
#         label = result['label']
#         confidence = result['confidence']
#         text = '{}: {:.0f}%'.format(label, confidence * 100)
#         frame = cv2.rectangle(frame, tl, br, color, 5)
#         frame = cv2.putText(frame, text, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)

#     cv2.imshow('frame', frame)
#     print('FPS {:.1f}'.format(1 / (time.time() - stime)))
#     if cv2.waitKey(1) == 27:
#         break

# capture.release()
# cv2.destroyAllWindows()


# while (capture.isOpened()):
#     stime = time.time()
#     ret, frame = capture.read()
#     if ret:
#         results = tfnet.return_predict(frame)
#         for color, result in zip(colors, results):
#             tl = (result['topleft']['x'], result['topleft']['y'])
#             br = (result['bottomright']['x'], result['bottomright']['y'])
#             label = result['label']
#             frame = cv2.rectangle(frame, tl, br, color, 7)
#             frame = cv2.putText(frame, label, tl, cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 2)
#         cv2.imshow('frame', frame)
#         print('FPS {:.1f}'.format(1 / (time.time() - stime)))
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#     else:
#         capture.release()
#         cv2.destroyAllWindows()
#         break
