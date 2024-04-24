## Robotic_Vision_SLAM
*Introduction to SLAM utilizing MATLAB as the exmaple code for the project.*

# Brief Summary to Implement SLAM (From MATLAB Ex)
There are a few methods that are being utilized here in the SLAM project. To begin, correspondence between frames is accounted for image comparison. The camera pose is then calculated to figure out the pose of the image being taken, then the image is triangulated so to output a 3D image map. 
To be more specific, the algorithm basically uses feature matching to determine homography. Once calculated, the relative camera pose is estimated, and triangulation begins. The final triangulation then is used to complete the 3D map of the surrounding area.  

To put what was stated previously in simple terms, SLAM utilizes several key methods to achieve simultaneous localization and mapping. It begins by establishing correspondences between frames through feature matching, allowing for image comparison. The camera pose is then computed to determine the position and orientation of the captured image. With this information, triangulation is performed to convert the 2D image data into a 3D map representation of the inputted image. The algorithm, ORB/SIFT, then employs feature matching to calculate homography, enabling the estimation of relative camera poses. This step utilizes triangulation, then results in the creation of a comprehensive 3D map of the surrounding environment. Once all these things are achieved, the SLAM algorithm should have the capability of running as intended for the desired output map. 

# Results From MATLAB Example
The below images show the expected output when running the exmaple code from the MATLAB link above. The individual data used to acquire this map can be optained [HERE](SLAM/MATLAB_Ex).


![image](https://github.com/RoboticsZ12/Robotic_Vision_SLAM/assets/142946153/54b7182c-0999-434e-a830-94fe52576636)

*Inputted image with feature matching enabled.*

![image](https://github.com/RoboticsZ12/Robotic_Vision_SLAM/assets/142946153/8583785b-0716-4287-af0a-26a787e778b0)

*Map created from inputted image.*

![image](https://github.com/RoboticsZ12/Robotic_Vision_SLAM/assets/142946153/afee9660-46dc-43e7-b8c4-5817123ab517)

*Map tracking image while moving.*

![image](https://github.com/RoboticsZ12/Robotic_Vision_SLAM/assets/142946153/d0f7b64e-a98d-4708-b5d5-124dbff19bf2)

*Final SLAM interpretation of the surrounding area.*

# Implementing SLAM
Implementing a SLAM project involves a series of critical steps, each step contributes to the system's accuracy and reliability of the system. Initially, we need to select a suitable framework or library, such as ORB-SLAM as described in the tutorial. This will provide a foundation for the project's development and overall mapping sequence. Alternatively, you could build the project from scratch using libraries and packages like OpenCV and Eigen. These packages allow for customization to specific project requirements. 

Next, gathering data must be acquired by gathering sensor data from cameras, LiDAR, or IMUs, ensuring synchronization for multiple sensors. We then implement feature matching, which employs algorithms like ORB, SIFT, or SURF to identify keypoints and establish correspondence between frames.  Once the data is accumulated, mapping of the area is done by utilizing estimated poses and triangulated points to construct an environment map, which varies based on SLAM type (e.g., visual or LiDAR). Loop closure detection identifies revisited locations to correct drift and enhance map consistency, followed by global map optimization to refine the entire map and enhance trajectory accuracy. The output of from my own images can be found [HERE](SLAM/MATLAB_Own_Implementation).

We were also requested to have one of our data photos to have us in it to prove it is our own data set. The image can be found [HERE](SLAM/MATLAB_Own_Implementation).
