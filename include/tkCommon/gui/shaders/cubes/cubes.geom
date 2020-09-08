///////////////////////////
//  1*-------------*2
//  |  \             \
//  |   3*------------*4
//  |   |              |
//  |   |              |
//  |   |              |
//  5*-------------*6  |
//     \|            \ |
//      7*------------*8
///////////////////////////

#version 330 core
layout (points) in;
layout (triangle_strip, max_vertices = 16) out;

uniform mat4 modelview;
out     vec3 fColor;

in VS_OUT {
    vec3 size;
    mat4 rotation;
    vec3 color;
}  gs_in[];

void main() {    

    fColor      = gs_in[0].color;
    vec4 center = vec4( gl_in[0].gl_Position.xyz, 1.0f);
    vec3 size   = gs_in[0].size/2;
    mat4 rot    = gs_in[0].rotation;

    vec4 vertex;

    //face-up
    /////////////////////////////////////////////////////////////////
    //1' vertex
    vertex = vec4(vec3(size.x, -size.y, size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //2' vertex
    vertex = vec4(vec3(size.x, size.y, size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //3' vertex
    vertex = vec4(vec3(-size.x, -size.y, size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //4' vertex
    vertex = vec4(vec3(-size.x, size.y, size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    //face-down
    /////////////////////////////////////////////////////////////////
    //7' vertex
    vertex = vec4(vec3(-size.x, -size.y, -size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //8' vertex
    vertex = vec4(vec3(-size.x, size.y, -size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //5' vertex
    vertex = vec4(vec3(size.x, -size.y, -size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //6' vertex
    vertex = vec4(vec3(size.x, size.y, -size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-left
    /////////////////////////////////////////////////////////////////
     //3' vertex
    vertex = vec4(vec3(-size.x, -size.y, size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //7' vertex
    vertex = vec4(vec3(-size.x, -size.y, -size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //1' vertex
    vertex = vec4(vec3(size.x, -size.y, size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //5' vertex
    vertex = vec4(vec3(size.x, -size.y, -size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    //face-left
    /////////////////////////////////////////////////////////////////
    //2' vertex
    vertex = vec4(vec3(size.x, size.y, size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //6' vertex
    vertex = vec4(vec3(size.x, size.y, -size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //4' vertex
    vertex = vec4(vec3(-size.x, size.y, size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //8' vertex
    vertex = vec4(vec3(-size.x, size.y, -size.z), 1.0f);
    vertex = vertex * rot;
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();
}