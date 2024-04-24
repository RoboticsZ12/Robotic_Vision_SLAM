# My Own Data

This portion consists of self-recorded data plugged into the sample code provided by MATLAB. There are some changes to the code that must be changed in order to have the code compile correctly, however the changes are not drastic. The first thing that had to be accomplished was the acquisition of data, or images. The example provided on MATLAB utilizes
approximately 150 images as its own data. 

Due to our objective being to just create the 3D map that has the "Estimated Trajectory" logged, I decided to input fewer images. I have uploaded approximately 50 images, which gave me the necessary output for the estimated trajectory 3D graph. However, as previously stated, I did need to change a few aspects of the code to have it compile correctly. The first of these changes was to comment out the URL download. 
For obvious reasons, we are using our own data, so we do not need to download their data. The commented out portion should appear as below. 
````
% baseDownloadURL = "https://vision.in.tum.de/rgbd/dataset/freiburg3/rgbd_dataset_freiburg3_long_office_household.tgz";
% dataFolder      = fullfile(tempdir, 'tum_rgbd_dataset', filesep);
% options         = weboptions(Timeout=Inf);
% tgzFileName     = [dataFolder, 'fr3_office.tgz'];
% folderExists    = exist(dataFolder, "dir");
%
% % Create a folder in a temporary directory to save the downloaded file
% if ~folderExists
%     mkdir(dataFolder);
%     disp('Downloading fr3_office.tgz (1.38 GB). This download can take a few minutes.')
%     websave(tgzFileName, baseDownloadURL, options);
%
%     % Extract contents of the downloaded file
%     disp('Extracting fr3_office.tgz (1.38 GB) ...')
%     untar(tgzFileName, dataFolder);
% end
````

As previously stated, this will cancel the download of the MATLAB data. Now, I must incorporate the data that I acquired to the MATLAB code. To do this, I need to change the "ImageFolder" variable to the designated location on my own device. The original code can be seen commented out, and the modified code with the modified path can be seen below.

````
% imageFolder = fullfile('C:\Users\zgeorgi\Documents\RB_images\RB\vision\');
imageFolder = fullfile('C:\Users\zgeorgi\Documents\RB_images_2\RB_2\vision\');
imds = imageDatastore(imageFolder);
% Inspect the first image
currFrameIdx = 1;
currI = readimage(imds, currFrameIdx);
himage = imshow(currI);
````




