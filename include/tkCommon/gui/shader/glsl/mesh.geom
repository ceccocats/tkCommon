#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 3) out;

uniform mat4 modelview;

out vec3 FragPos;
out vec3 Normal;

in VS_OUT {
    vec3 point1;
    vec3 normal1;
    vec3 point2;
    vec3 normal2;
    vec3 point3;
    vec3 normal3;
} gs_in[];

void main() {    

    vec4 point1     = vec4(gs_in[0].point1, 1.0);
    vec3 normal1    = gs_in[0].normal1;
    vec4 point2     = vec4(gs_in[0].point2, 1.0);
    vec3 normal2    = gs_in[0].normal2;
    vec4 point3     = vec4(gs_in[0].point3, 1.0);
    vec3 normal3    = gs_in[0].normal3;

    //1' vertex
    gl_Position = modelview * point1;
    FragPos     = point1.xyz;
    Normal      = normal1;
    EmitVertex();

    //2' vertex
    gl_Position = modelview * point2;
    FragPos     = point2.xyz;
    Normal      = normal2;
    EmitVertex();

    //3' vertex
    gl_Position = modelview * point3;
    FragPos     = point3.xyz;
    Normal      = normal3;
    EmitVertex();

    EndPrimitive();
}