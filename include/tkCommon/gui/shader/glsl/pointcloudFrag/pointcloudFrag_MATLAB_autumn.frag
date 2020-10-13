// this file is generated DO NOT DIRECTLY MODIFY
#version 330 core

#include "../colormaps/MATLAB_autumn.frag"

out vec4 FragColor;

in float feature;

uniform float minFeature;
uniform float maxFeature;

void main(){
	float value	= (feature - minFeature) / (maxFeature - minFeature);
	FragColor	= colormap(value);
}