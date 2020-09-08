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

uniform mat4    modelview;
uniform float   size;

out vec4 point;

void main() {    

    vec4 center = vec4( gl_in[0].gl_Position.xyz, 1.0f);
    float dim   = size / 2;

    vec4 vertex;

    point = center;

    //face-up
    /////////////////////////////////////////////////////////////////
    //1' vertex
    vertex = vec4(vec3(dim, -dim, dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //2' vertex
    vertex = vec4(vec3(dim, dim, dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //3' vertex
    vertex = vec4(vec3(-dim, -dim, dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //4' vertex
    vertex = vec4(vec3(-dim, dim, dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    //face-down
    /////////////////////////////////////////////////////////////////
    //7' vertex
    vertex = vec4(vec3(-dim, -dim, -dim), 1.0f);
    
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //8' vertex
    vertex = vec4(vec3(-dim, dim, -dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //5' vertex
    vertex = vec4(vec3(dim, -dim, -dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //6' vertex
    vertex = vec4(vec3(dim, dim, -dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-left
    /////////////////////////////////////////////////////////////////
     //3' vertex
    vertex = vec4(vec3(-dim, -dim, dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //7' vertex
    vertex = vec4(vec3(-dim, -dim, -dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //1' vertex
    vertex = vec4(vec3(dim, -dim, dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //5' vertex
    vertex = vec4(vec3(dim, -dim, -dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    //face-left
    /////////////////////////////////////////////////////////////////
    //2' vertex
    vertex = vec4(vec3(dim, dim, dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //6' vertex
    vertex = vec4(vec3(dim, dim, -dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //4' vertex
    vertex = vec4(vec3(-dim, dim, dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    //8' vertex
    vertex = vec4(vec3(-dim, dim, -dim), 1.0f);
    vertex = vertex + center;
    gl_Position = modelview * vertex;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();
}