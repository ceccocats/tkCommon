#version 330 core
layout (points) in;
layout (line_strip, max_vertices = 6) out;

uniform mat4 modelview;

out vec3 fColor;

void main() {    

    vec4 center     = vec4(gl_in[0].gl_Position.xyz, 1.0f);
    float lenght    = gl_in[0].gl_Position.w;

    vec4 vertex;
    mat4 rotation;

    rotation[0] = vec4(modelview[0][0], modelview[0][1],    modelview[0][2],   0.0f);
    rotation[1] = vec4(modelview[1][0], modelview[1][1],    modelview[1][2],   0.0f);
    rotation[2] = vec4(modelview[2][0], modelview[2][1],    modelview[2][2],   0.0f);
    rotation[3] = vec4(0.0f,            0.0f,               0.0f,              1.0f);
    
    // x-axis
    fColor      = vec3(1.0f, 0.2f, 0.2f);
    gl_Position = center;
    EmitVertex();
    gl_Position = (rotation * vec4(lenght, 0.0f, 0.0f, 0.0f)) + center;
    EmitVertex();

    EndPrimitive();

    // y-axis
    fColor      = vec3(0.2f, 1.0f, 0.2f);
    gl_Position = center;
    EmitVertex();
    gl_Position = (rotation * vec4(0.0f, lenght, 0.0f, 0.0f)) + center;
    EmitVertex();

    EndPrimitive();

    // z-axis
    fColor      = vec3(0.2f, 0.2f, 1.0f);
    gl_Position = center;
    EmitVertex();
    gl_Position = (rotation * vec4(0.0f, 0.0f, lenght, 0.0f)) + center;
    EmitVertex();

    EndPrimitive();
}