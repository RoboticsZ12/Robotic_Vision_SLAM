# MATLAB EXAMPLE
A described in the README, this data and code was used and can be found as an example from the MATLAB website provided below.

[Monocular Visual Simultaneous Localization and Mapping (SLAM)](https://www.mathworks.com/help/vision/ug/monocular-visual-simultaneous-localization-and-mapping.html)

This example code, when compiled, will download all external image data from the needed website. This will then begin the SLAM process of the algorithm. First, the algorithm utilizes two images at a time for comparison. By comaring two images at a time, the algorithm utilizes SIFT/SURF algorithms to detect any and all the corners located within the two images. 
By doing this, the algorithm can calculate a 3D map of the surrounding area.

Something to keep in mind when compiling your own data will be the amount of photos you take for the 3D mapping. When the program finishes compiling, the algorithm will basically create a "movie" type map of your images to capture all the necessary points. This is the reasoning of ensuring you have a great many photos per cm of movement of the camera. If the camera is moved to fast between point A and point B, then the image comparison will be unable to properly detect the corners of the provided images. For this reason, the example data has a multitude of photos to ensure the algorithm outputs in the desired manner. 

After compiling the entire program, the algorithm should output a 3D map of the area you decided to interract with as shown below. 

![image](https://github.com/RoboticsZ12/Robotic_Vision_SLAM/assets/142946153/d0f7b64e-a98d-4708-b5d5-124dbff19bf2)

It can clearly be seen in the image that there are three seperate data plots on the single graph. These plots represent the ground truth data, or the path that was acutally traveled, an estimated path that the compiler interpreted the camera user took, and finally an optimal path, or the path that should have been taken for the best data possible. 
