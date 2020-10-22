#version 330 core

const float lenght  =  0.04f;
uniform int width;
uniform int height;

void main()
{
    float cx    = -1.0f + 150.0f/width;
    float cy    = -1.0f + 125.0f/height;
    gl_Position = vec4(cx, cy, 0.0f, lenght);
}

    