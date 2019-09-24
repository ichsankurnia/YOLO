### Training on your own dataset

*The steps below assume we want to use tiny YOLO and our dataset has 3 classes*

1. Create a copy of the configuration file `tiny-yolo-voc.cfg` and rename it according to your preference `tiny-yolo-voc-3c.cfg` (It is crucial that you leave the original `tiny-yolo-voc.cfg` file unchanged, see below for explanation).

2. In `tiny-yolo-voc-3c.cfg`, change classes in the [region] layer (the last layer) to the number of classes you are going to train for. In our case, classes are set to 3.
    
    ```python
    ...

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    bias_match=1
    classes=3
    coords=4
    num=5
    softmax=1
    
    ...
    ```

3. In `tiny-yolo-voc-3c.cfg`, change filters in the [convolutional] layer (the second to last layer) to num * (classes + 5). In our case, num is 5 and classes are 3 so 5 * (3 + 5) = 40 therefore filters are set to 40.
    
    ```python
    ...

    [convolutional]
    size=1
    stride=1
    pad=1
    filters=40
    activation=linear

    [region]
    anchors = 1.08,1.19,  3.42,4.41,  6.63,11.38,  9.42,5.11,  16.62,10.52
    
    ...
    ```

4. Change `labels.txt` to include the label(s) you want to train on (number of labels should be the same as the number of classes you set in `tiny-yolo-voc-3c.cfg` file). In our case, `labels.txt` will contain 3 labels.

    ```
    label1
    label2
    label3
    ```
5. Reference the `tiny-yolo-voc-3c.cfg` model when you train.

    `flow --model cfg/tiny-yolo-voc-3c.cfg --load bin/tiny-yolo-voc.weights --train --annotation train/Annotations --dataset train/Images`


* Why should I leave the original `tiny-yolo-voc.cfg` file unchanged?
    
    When darkflow sees you are loading `tiny-yolo-voc.weights` it will look for `tiny-yolo-voc.cfg` in your cfg/ folder and compare that configuration file to the new one you have set with `--model cfg/tiny-yolo-voc-3c.cfg`. In this case, every layer will have the same exact number of weights except for the last two, so it will load the weights into all layers up to the last two because they now contain different number of weights.


## Camera/video file demo

For a demo that entirely runs on the CPU:

```bash
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi
```

For a demo that runs 100% on the GPU:

```bash
flow --model cfg/yolo-new.cfg --load bin/yolo-new.weights --demo videofile.avi --gpu 1.0
```

To use your webcam/camera, simply replace `videofile.avi` with keyword `camera`.

To save a video with predicted bounding box, add `--saveVideo` option.

## Using darkflow from another python application

Please note that `return_predict(img)` must take an `numpy.ndarray`. Your image must be loaded beforehand and passed to `return_predict(img)`. Passing the file path won't work.

Result from `return_predict(img)` will be a list of dictionaries representing each detected object's values in the same format as the JSON output listed above.

```python
from darkflow.net.build import TFNet
import cv2

options = {"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold": 0.1}

tfnet = TFNet(options)

imgcv = cv2.imread("./sample_img/sample_dog.jpg")
result = tfnet.return_predict(imgcv)
print(result)
```


## Save the built graph to a protobuf file (`.pb`)

```bash
## Saving the lastest checkpoint to protobuf file
flow --model cfg/yolo-new.cfg --load -1 --savepb

## Saving graph and weights to protobuf file
flow --model cfg/yolo.cfg --load bin/yolo.weights --savepb
```
When saving the `.pb` file, a `.meta` file will also be generated alongside it. This `.meta` file is a JSON dump of everything in the `meta` dictionary that contains information nessecary for post-processing such as `anchors` and `labels`. This way, everything you need to make predictions from the graph and do post processing is contained in those two files - no need to have the `.cfg` or any labels file tagging along.

The created `.pb` file can be used to migrate the graph to mobile devices (JAVA / C++ / Objective-C++). The name of input tensor and output tensor are respectively `'input'` and `'output'`. For further usage of this protobuf file, please refer to the official documentation of `Tensorflow` on C++ API [_here_](https://www.tensorflow.org/versions/r0.9/api_docs/cc/index.html). To run it on, say, iOS application, simply add the file to Bundle Resources and update the path to this file inside source code.

Also, darkflow supports loading from a `.pb` and `.meta` file for generating predictions (instead of loading from a `.cfg` and checkpoint or `.weights`).
```bash
## Forward images in sample_img for predictions based on protobuf file
flow --pbLoad built_graph/yolo.pb --metaLoad built_graph/yolo.meta --imgdir sample_img/
```
If you'd like to load a `.pb` and `.meta` file when using `return_predict()` you can set the `"pbLoad"` and `"metaLoad"` options in place of the `"model"` and `"load"` options you would normally set.
