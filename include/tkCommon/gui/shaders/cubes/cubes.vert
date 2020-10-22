#version 330 core
layout (location = 0) in vec3 center;
layout (location = 1) in vec3 size;
layout (location = 2) in vec3 rot;
layout (location = 3) in vec3 color;

out VS_OUT {
    vec3 size;
    mat4 rotation;
    vec3 color;
} vs_out;

void main()
{
    vs_out.size     = size;
    vs_out.color    = color;
    vs_out.rotation = mat4(
        cos(rot.z)*cos(rot.y),  cos(rot.z)*sin(rot.y)*sin(rot.x) - sin(rot.z)*cos(rot.x),   cos(rot.z)*sin(rot.y)*cos(rot.x) + sin(rot.z)*sin(rot.x),   0,
        sin(rot.z)*cos(rot.y),  sin(rot.z)*sin(rot.y)*sin(rot.x) + cos(rot.z)*cos(rot.x),   sin(rot.z)*sin(rot.y)*cos(rot.x) - cos(rot.z)*sin(rot.x),   0,
        -sin(rot.y),            cos(rot.y)*sin(rot.x),                                      cos(rot.y)*cos(rot.x),                                      0,
        0,                      0,                                                          0,                                                          1                               
    );
    gl_Position     = vec4(center.xyz, 1.0);
}

    