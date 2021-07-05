# tkCommon
## Build
#### install deps
```
# cmake from source
cmake >= 3.18: https://cmake.org/download/

# => from apt:
# basic stuff to build
sudo apt-get install git build-essential  
# graphics
sudo apt-get install libgles2-mesa-dev    # NOT on aarch64
sudo apt-get install libglew-dev libglfw3-dev freeglut3-dev libfreetype6-dev libglm-dev
# misc
sudo apt-get install libeigen3-dev libyaml-cpp-dev libpcap-dev libmatio-dev libgeographic-dev libpugixml-dev libudev-dev python3-dev 

# optional: 
- cuda
- ROS melodic with this packages: ros-melodic-desktop ros-melodic-tf2-geometry-msgs ros-melodic-ackermann-msgs
- LibSerial: https://github.com/FisherTiger95/libserial
- Lanelet2: https://github.com/FisherTiger95/lanelet2_standalone
```
#### compile
```
mkdir build
cd build 
cmake .. -DCMAKE_BUILD_TYPE=RelWithDebInfo
make -j8
```

## Third party software
| Repository | Author | LICENSE |
|------------|--------|---------|
|[can-dbcparser](https://github.com/downtimes/can-dbcparser) |Michael Auracher | MIT|
|[imgui](https://github.com/ocornut/imgui)| Omar Cornut | MIT|
|[implot](https://github.com/epezent/implot)| Evan Pezent | MIT|
|[stb_image](https://github.com/nothings/stb)|Michael Keck| MIT|
|[OBJ-Loader](https://github.com/Bly7/OBJ-Loader)|Robert Smith| MIT|
|[Argh!](https://github.com/adishavit/argh)| Adi Shavit| Argh! |
|[geodetic_utils](https://github.com/ethz-asl/geodetic_utils)| Enric Galceran, Marija Popović |BSD3|
|[gamepad](https://github.com/elanthis/gamepad)| seanmiddleditch | MIT|
|[colormap-shaders](https://github.com/kbinani/colormap-shaders)| kbinani | MIT|

