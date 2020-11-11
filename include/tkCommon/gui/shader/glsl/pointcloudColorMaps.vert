#version 330 core

layout (location = 0) in vec4 point;
layout (location = 1) in float featurePoint;

uniform mat4 modelview;

out float feature;

uniform int axis;

void main(){

    gl_Position 	= modelview * vec4(point.xyz, 1.0);

    if(axis == -1){
        feature = featurePoint;
        return;
    }

    if(axis == 0){
        feature = point.x;
        return;
    }

    if(axis == 1){
        feature = point.y;
        return;
    }

    if(axis == 2){
        feature = point.z;
        return;
    }
    
}
