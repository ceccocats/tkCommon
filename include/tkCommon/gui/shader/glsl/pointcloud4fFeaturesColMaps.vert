#version 330 core

layout (location = 0) in vec4 point;
layout (location = 1) in float featurePoint;

uniform mat4 modelview;

out float feature;

void main(){

    feature    		= featurePoint;
	gl_Position 	= modelview * vec4(point.xyz, 1.0);
}
