#version 330 core
layout (location = 0) in vec4 points;

void main()
{
    gl_Position = vec4(points.xyz, 1.0);
}

    