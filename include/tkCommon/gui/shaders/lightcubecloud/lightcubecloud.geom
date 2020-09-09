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
layout (triangle_strip, max_vertices = 24) out;

uniform mat4    modelview;
uniform float   size;

out vec3 FragPos;
out vec3 Normal;

vec3 facenormals[6] = vec3[](
            vec3(0.0f, 0.0f, 1.0f),
            vec3(0.0f, 0.0f, -1.0f),
            vec3(-1.0f, 0.0f, 0.0f),
            vec3(1.0f, 0.0f, 0.0f),
            vec3(0.0f, -1.0f, 0.0f),
            vec3(0.0f, 1.0f, 0.0f)
);

void main() {    

    vec4 center = vec4( gl_in[0].gl_Position.xyz, 1.0f);
    float dim   = size / 2;

    vec4 vertex;

    //face-up
    /////////////////////////////////////////////////////////////////
    Normal      = facenormals[0];

    //1' vertex
    vertex      = vec4(vec3(dim, -dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //2' vertex
    vertex      = vec4(vec3(dim, dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //3' vertex
    vertex      = vec4(vec3(-dim, -dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //4' vertex
    vertex      = vec4(vec3(-dim, dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-down
    /////////////////////////////////////////////////////////////////
    Normal      = facenormals[1];
    
    //5' vertex
    vertex      = vec4(vec3(dim, -dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //6' vertex
    vertex      = vec4(vec3(dim, dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //7' vertex
    vertex      = vec4(vec3(-dim, -dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //8' vertex
    vertex      = vec4(vec3(-dim, dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-front
    /////////////////////////////////////////////////////////////////
    Normal      = facenormals[2];

    //3' vertex
    vertex      = vec4(vec3(-dim, -dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //4' vertex
    vertex      = vec4(vec3(-dim, dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //7' vertex
    vertex      = vec4(vec3(-dim, -dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //8' vertex
    vertex      = vec4(vec3(-dim, dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-back
    /////////////////////////////////////////////////////////////////
    Normal      = facenormals[3];

    //1' vertex
    vertex      = vec4(vec3(dim, -dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //2' vertex
    vertex      = vec4(vec3(dim, dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //5' vertex
    vertex      = vec4(vec3(dim, -dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //6' vertex
    vertex      = vec4(vec3(dim, dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-left
    /////////////////////////////////////////////////////////////////
    Normal      = facenormals[4];

     //3' vertex
    vertex      = vec4(vec3(-dim, -dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //7' vertex
    vertex      = vec4(vec3(-dim, -dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //1' vertex
    vertex      = vec4(vec3(dim, -dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //5' vertex
    vertex      = vec4(vec3(dim, -dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-left
    /////////////////////////////////////////////////////////////////
    Normal      = facenormals[5];

    //2' vertex
    vertex = vec4(vec3(dim, dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //6' vertex
    vertex = vec4(vec3(dim, dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //4' vertex
    vertex = vec4(vec3(-dim, dim, dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    //8' vertex
    vertex = vec4(vec3(-dim, dim, -dim), 1.0f) + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();
}