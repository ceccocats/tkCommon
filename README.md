# tkSlam
All-Around slam library.

## WARNING
The library is in early development stage, some important tasks related to 
PointClouds are delegated to the **point cloud library**. 
Since the goal is to suppress this dependency the main data structure is
not related to this dependency sacrificing performance for
less effort in future replacing.

## Prerequisites
* [Ubuntu 16.04.5](http://releases.ubuntu.com/16.04/)
* [g2o](https://github.com/RainerKuemmerle/g2o) version: 20160424_git
* [eigen 3.3.5](http://bitbucket.org/eigen/eigen/get/3.3.5.tar.bz2)
* [pcl](https://github.com/PointCloudLibrary/pcl)
* ffmpeg as a command line utils

## Build
```bash
mkdir build
cd build
cmake .. # this will downlad a test dataset
make -j
```

## Dataset format
Currently the software accept datasets as form of directory with 2 file for every laserscan:
- cloudN.txt
- odomN.txt
where N is a number for every frame (starting from zero)

cloudN.txt format:
```
x0 y0 z0
x1 y1 z1
x2 y2 z2
...
```
odomN.txt format:
```
x y yaw timestamp
gps_x gps_y gps_z  # considered only if gps is enable in conf file
```

## Dataset from rosbag
Its possible to convert a rosbag to the accepted dataset using the rosbag2txt.py utility:
```bash
# usage: 
python ../test/ros2txt.py rosbag PointCloud2_topic Odom_topic outdir

# example
python ../test/ros2txt.py rec.bag /points2 /odom rec/
```


## Test
#### Align
Test a simple PointCloud alignment:
```bash
# this will use dataset data
./test_align  
# you can provide 2 cloud to align (and optionally odometry)
./test_align cloudA.txt cloudB.txt <odomA.txt> <odomB.txt>
```

#### SLAM
Test SLAM:
```bash
# this will use dataset data
./test_slam  
# you can provide differents datasets
./test_slam path/to/dataset
```
Press Esc to finish and save the map, it will save it in the "map" directory

**Notes:**<br> 
The SLAM must be properly configured based on the environment the robot
is exploring and the precision of the Odometry.
Configurations are on top of *test_slam.cpp*:
```c++
// confs
float ODOM_THRESH = 3.0;                // meters to trigger a scan record
float LOOP_THRESH = 1.0;                // max score to add a loop connection
float LOOP_DIST_THRESH = 10.0;          // max dist to search for a loop
float LOOP_ACCUM_DIST_THRESH = 20.0;    // min travelled dist for loop search 
```
Tips:
- Small enviroments will needs smaller ODOM_THRESH, a value of 1.0 means a new Keyframe every meter.
- LOOP_THRESH is the acceptable quality of a loop detection, smaller values will increase the precision but doesn't guarantees to find a connection
- Big map with Odometry drift need bigger LOOP_DIST_THRESH and LOOP_ACCUM_DIST_THRESH


#### Localization
Test localization:
```bash
# this will use dataset data
./test_loc 
# you can provide differents datasets
./test_loc path/to/dataset path/to/map
```

# GEOREF map
gdal is an utility for georef a map this is how it works:
```bash
gdal_translate -of GTiff -gcp x0 y0 lat0 lon0 -gcp x1 y1 lat1 lon1 ... "map.pgm" "/tmp/map.pgm"
gdalwarp -r near -order 1 -co COMPRESS=NONE  "/tmp/map.pgm" "map.tif"
```


## Authors

* **Gatti Francesco** - *Main developer* - [fgatti](https://git.hipert.unimore.it/fgatti)
* **Bosi Massimiliano** - *Developer* - [mbosi](https://git.hipert.unimore.it/mbosi)
