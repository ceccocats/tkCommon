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

uniform mat4 modelview;

out vec3 FragPos;
out vec3 Normal;
out vec3 objectColor;

in VS_OUT {
    vec3 size;
    mat4 rotation;
    vec3 color;
}  gs_in[];

vec3 facenormals[6] = vec3[](
            vec3(0.0f, 0.0f, 1.0f),
            vec3(0.0f, 0.0f, -1.0f),
            vec3(-1.0f, 0.0f, 0.0f),
            vec3(1.0f, 0.0f, 0.0f),
            vec3(0.0f, -1.0f, 0.0f),
            vec3(0.0f, 1.0f, 0.0f)
);

void main() {    

    objectColor = gs_in[0].color;
    vec4 center = vec4( gl_in[0].gl_Position.xyz, 1.0f);
    vec3 size   = gs_in[0].size/2;
    mat4 rot    = gs_in[0].rotation;

    vec4 vertex;

    //face-up
    /////////////////////////////////////////////////////////////////
    //1' vertex
    vertex      = vec4(vec3(size.x, -size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[0];
    EmitVertex();
    //2' vertex
    vertex = vec4(vec3(size.x, size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[0];
    EmitVertex();
    //3' vertex
    vertex = vec4(vec3(-size.x, -size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[0];
    EmitVertex();
    //4' vertex
    vertex      = vec4(vec3(-size.x, size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[0];
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-down
    /////////////////////////////////////////////////////////////////
    //5' vertex
    vertex      = vec4(vec3(size.x, -size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[1];
    EmitVertex();
    //6' vertex
    vertex      = vec4(vec3(size.x, size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[1];
    EmitVertex();
    //7' vertex
    vertex      = vec4(vec3(-size.x, -size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[1];
    EmitVertex();
    //8' vertex
    vertex      = vec4(vec3(-size.x, size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[1];
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-front
    /////////////////////////////////////////////////////////////////
    //3' vertex
    vertex      = vec4(vec3(-size.x, -size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[2];
    EmitVertex();
    //4' vertex
    vertex      = vec4(vec3(-size.x, size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[2];
    EmitVertex();
    //7' vertex
    vertex      = vec4(vec3(-size.x, -size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[2];
    EmitVertex();
    //8' vertex
    vertex      = vec4(vec3(-size.x, size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[2];
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-back
    /////////////////////////////////////////////////////////////////
    //1' vertex
    vertex      = vec4(vec3(size.x, -size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[3];
    EmitVertex();
    //2' vertex
    vertex      = vec4(vec3(size.x, size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[3];
    EmitVertex();
    //5' vertex
    vertex      = vec4(vec3(size.x, -size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[3];
    EmitVertex();
    //6' vertex
    vertex      = vec4(vec3(size.x, size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[3];
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-left
    /////////////////////////////////////////////////////////////////
     //3' vertex
    vertex      = vec4(vec3(-size.x, -size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[4];
    EmitVertex();
    //7' vertex
    vertex      = vec4(vec3(-size.x, -size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[4];
    EmitVertex();
    //1' vertex
    vertex      = vec4(vec3(size.x, -size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[4];
    EmitVertex();
    //5' vertex
    vertex      = vec4(vec3(size.x, -size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[4];
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();

    //face-right
    /////////////////////////////////////////////////////////////////
    //2' vertex
    vertex      = vec4(vec3(size.x, size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[5];
    EmitVertex();
    //6' vertex
    vertex      = vec4(vec3(size.x, size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[5];
    EmitVertex();
    //4' vertex
    vertex      = vec4(vec3(-size.x, size.y, size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[5];
    EmitVertex();
    //8' vertex
    vertex      = vec4(vec3(-size.x, size.y, -size.z), 1.0f) * rot + center;
    gl_Position = modelview * vertex;
    FragPos     = vertex.xyz;
    Normal      = mat3(transpose(rot)) * facenormals[5];
    EmitVertex();
    /////////////////////////////////////////////////////////////////

    EndPrimitive();
}