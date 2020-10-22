#version 330 core

#include "../colormaps/MATLAB_hsv.frag"

out vec4 FragColor;

in vec4 point;

const float range = 20.0f;

void main(){

	FragColor = colormap(mod(point.z,range)/range);
}