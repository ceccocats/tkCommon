#version 330 core
layout (location = 0) in vec3 point1;
layout (location = 1) in vec3 normal1;
layout (location = 2) in vec3 point2;
layout (location = 3) in vec3 normal2;
layout (location = 4) in vec3 point3;
layout (location = 5) in vec3 normal3;

out VS_OUT {
    vec3 point1;
    vec3 normal1;
    vec3 point2;
    vec3 normal2;
    vec3 point3;
    vec3 normal3;
} vs_out;

void main()
{
    vs_out.point1   = point1;    
    vs_out.normal1  = normal1;
    vs_out.point2   = point2;
    vs_out.normal2  = normal2;
    vs_out.point3   = point3;
    vs_out.normal3  = normal3;   
}
