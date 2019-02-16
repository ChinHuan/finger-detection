# finger-detection
This script is developed using Python 2.7 and OpenCV 3.2.0

## FingersNumberDetector.py
### Usage
1. Run the script by
```
python FingersNumberDetector.py
```
2. Place your hand on the 9 green squares and capture your skin color by pressing *z*.
3. Move the ROI (blue square) if you want.
4. Move your hand away (any moving objects) from the ROI.
5. Capture the background by pressing *b*.
6. Now you can start placing your hand in the ROI.

### Shortcuts
Key | Description
--- | ---
j | Move ROI Down
k | Move ROI Up
h | Move ROI Left
l | Move ROI Right
q | Exit
b | Capture the background
r | Reset the background subtractor

## CNN-FingersNumberDetector.py
Basically, it is the same thing as FingersNumberDetector.py. The only difference is that CNN is used for counting the number of fingers.

### Usage
Same as above with additional functionality to sample training and testing data.
Remember to create the directories below before using.
```
\dataset\
|-- \train\
|-- \test\
```

### Shortcuts
Same as above with some additional keys below.

Key | Description
--- | ---
u | Sample training data
i | Sample testing data
0 | Sample "zero" image
1 | Sample "one" image
2 | Sample "two" image
3 | Sample "three" image
4 | Sample "four" image
5 | Sample "five" image



## References
1. lzane/Fingers-Detection-using-OpenCV-and-Python. https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python
2. amarlearning/Finger-Detection-and-Tracking. https://github.com/amarlearning/Finger-Detection-and-Tracking
