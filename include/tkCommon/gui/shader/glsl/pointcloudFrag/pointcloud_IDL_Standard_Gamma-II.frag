// this file is generated DO NOT DIRECTLY MODIFY
#version 330 core

#include "../colormaps/IDL_Standard_Gamma-II.frag"

out vec4 FragColor;

in float feature;

uniform float alpha;
uniform float minFeature;
uniform float maxFeature;

void main(){
	float value	= (feature - minFeature) / (maxFeature - minFeature);
	FragColor	= colormap(value);
	FragColor.a	= alpha;
}
