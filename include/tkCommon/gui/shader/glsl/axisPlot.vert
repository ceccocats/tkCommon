#version 330 core

layout (location = 0) in vec3 point;
layout (location = 1) in vec3 rot;

const float lenght  =  1.0f;

out VS_OUT {
    vec3 rotPt;
}  vs_out;

void main()
{
    vs_out.rotPt = rot;
    gl_Position = vec4(point.x, point.y, point.z, lenght);
}

    