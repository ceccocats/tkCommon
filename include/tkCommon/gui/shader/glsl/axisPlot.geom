#version 330 core
layout (points) in;
layout (line_strip, max_vertices = 6) out;

uniform mat4 modelview;

in VS_OUT {
    vec3 rotPt;
}  gs_in[];

out vec3 fColor;

void main() {    

    vec4 center     = vec4(gl_in[0].gl_Position.xyz, 1.0f);
    float lenght    = gl_in[0].gl_Position.w;

    vec3 rot = gs_in[0].rotPt;
    mat4 rotation = mat4(
        cos(rot.z)*cos(rot.y),  cos(rot.z)*sin(rot.y)*sin(rot.x) - sin(rot.z)*cos(rot.x),   cos(rot.z)*sin(rot.y)*cos(rot.x) + sin(rot.z)*sin(rot.x),   0,
        sin(rot.z)*cos(rot.y),  sin(rot.z)*sin(rot.y)*sin(rot.x) + cos(rot.z)*cos(rot.x),   sin(rot.z)*sin(rot.y)*cos(rot.x) - cos(rot.z)*sin(rot.x),   0,
        -sin(rot.y),            cos(rot.y)*sin(rot.x),                                      cos(rot.y)*cos(rot.x),                                      0,
        0,                      0,                                                          0,                                                          1                               
    );

    rotation = rotation ;//* modelview;
    
    // x-axis
    fColor      = vec3(1.0f, 0.3f, 0.3f);
    gl_Position = rotation * center;
    EmitVertex();
    gl_Position = (rotation * vec4(lenght, 0.0f, 0.0f, 0.0f)) + center;
    EmitVertex();

    EndPrimitive();

    // y-axis
    fColor      = vec3(0.3f, 1.0f, 0.3f);
    gl_Position = rotation * center;
    EmitVertex();
    gl_Position = (rotation * vec4(0.0f, lenght, 0.0f, 0.0f)) + center;
    EmitVertex();

    EndPrimitive();

    // z-axis
    fColor      = vec3(0.3f, 0.3f, 1.0f);
    gl_Position = rotation * center;
    EmitVertex();
    gl_Position = (rotation * vec4(0.0f, 0.0f, lenght, 0.0f)) + center;
    EmitVertex();

    EndPrimitive();
}