#version 330 core
layout (location = 0) in vec3 point;
layout (location = 1) in vec3 color;

out vec4 Fragcolor;

uniform mat4 modelview;

void main(){
    gl_Position = modelview * vec4(point, 1.0);
    Fragcolor   = vec4(color,1.0);
}