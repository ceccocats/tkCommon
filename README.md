# tkCommon
## Build
#### install deps
```
sudo apt-get install libgles2-mesa-dev # NOT on aarch64
sudo apt install git build-essential cmake rsync libeigen3-dev libglew-dev libglfw3-dev freeglut3-dev libfreetype6-dev libyaml-cpp-dev libpcap-dev libmatio-dev
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
|[libdrawtext](https://github.com/jtsiomb/libdrawtext) | John Tsiombikas |LGPL-3.0|
|[stb_image](https://github.com/nothings/stb)|Michael Keck| MIT|
|[OBJ-Loader](https://github.com/Bly7/OBJ-Loader)|Robert Smith| MIT|
|[Argh!](https://github.com/adishavit/argh)| Adi Shavit| Argh! |
|[geodetic_utils](https://github.com/ethz-asl/geodetic_utils)| Enric Galceran, Marija Popović |BSD3|

