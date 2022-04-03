# Pupil Detection Using Opencv


#### Description:

This is a Pupil Detector created by using `OpenCV` and `mediapipe`.

This uses the Haar-cascade detection algorithm to first detect faces and then
based on the detected face it detects individual eyes. Pupil detection is done by
using the `BlobDetection` algorithm. A blob is a group of connected pixels in an image that shares
common properties. 

There are different parameters of `BlobDetection` and in this program, we can select
the area, circularity, inertia, and convexity of the detector.
![img.png](https://raw.githubusercontent.com/FatihcDeniz/Artifical-Intelligence-and-Machine-Learning-Projects/main/Pupil%20Detector/Images/BlobTest.webp?token=GHSAT0AAAAAABR5FUENO455TFBHAP43JGFGYSJXGIQ)

Area parameter filters blobs based on their size. Circularity measures
how close to a circle the blob is. Inertia measures how elongated the shape 
of the blob is and convexity is the difference between the area of the blob and the area of its
convex hull.

Eye detection is done by applying multiple steps. The first step is detecting eyes
using haar-cascades. After detection, we change the color of eye images to gray,
then we apply a `binary threshold` to each eye. This will help us to detect
pupils because of the transformation with an appropriate threshold rate only pupil and eyebrows 
become `0` all other pixels will become `1`. It is really important to select an appropriate
threshold value while applying the Binary Threshold to the image. Values below the threshold become
`0` and values bigger than the threshold become `1`. In this program, we are free to increase or decrease
the threshold based on the lighting in the video. There are also other image processing techniques
we use to make pupils distinguishable in the picture. The first one is `Erosion`. `Erosion` is used to
remove white noises from the image, but this also makes our object smaller. Because of that, we 
use `Dilation` to increase the area of our object. Finally, we apply `MedianBlur` to smoothen the image and make
Pupil more detectable.

This picture shows original eye images:

![image](https://raw.githubusercontent.com/FatihcDeniz/Artifical-Intelligence-and-Machine-Learning-Projects/main/Pupil%20Detector/Images/Screen%20Shot%202022-04-03%20at%2015.05.53.png?token=GHSAT0AAAAAABR5FUEMCLN456OILHFI6YKSYSJXF6A)

and this shows how eye images look after binary transformations and other processing techniques.

![](https://raw.githubusercontent.com/FatihcDeniz/Artifical-Intelligence-and-Machine-Learning-Projects/main/Pupil%20Detector/Images/Screen%20Shot%202022-04-03%20at%2015.06.02.png?token=GHSAT0AAAAAABR5FUEMLPCWFLL3KO2LLM5AYSJXGWQ)

It is also possible to visualize Pupils by using contours. Contours are simply curves joining
all the continuous points having the same color and intensity. 

![](https://github.com/FatihcDeniz/Artifical-Intelligence-and-Machine-Learning-Projects/blob/main/Pupil%20Detector/Images/Screen%20Shot%202022-04-03%20at%2017.02.15.png?raw=true)

This program also allows us to show `face mesh` using `mediapipe` library. 

![](https://raw.githubusercontent.com/FatihcDeniz/Artifical-Intelligence-and-Machine-Learning-Projects/main/Pupil%20Detector/Images/Screen%20Shot%202022-04-03%20at%2017.01.52.png?token=GHSAT0AAAAAABR5FUEM2N3TMIACV63JABDWYSJXG7Q)

And finally, it allows us to detect head position. This is done by using
landmarks in the `media pipe` library. Based on the nose landmark location 
in the video we project a point to estimate where it is located.

#### Description:

`OpenCV`, `mediapipe`, `PIL` and `tkinter` libraries are required.

#### How to run:

You can run this program by writing `python eye-tracking.py` in the terminal.

References:
- https://opencv24-python-tutorials.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
- https://learnopencv.com/blob-detection-using-opencv-python-c/
- https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
