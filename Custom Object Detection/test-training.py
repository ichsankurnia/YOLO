import cv2
from darkflow.net.build import TFNet
import numpy as np
import time

option = {
    'model': 'cfg/tiny-yolo-voc-1c.cfg',
    'load': 125,
    'threshold': 0.15,
    # 'gpu' : 0.8
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