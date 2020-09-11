#version 330 core

uniform mat4 modelview;
uniform int n;
uniform int dir;
uniform float dim;

out vec4 objectColor;

void main()
{
    float sign = (gl_VertexID %2)*2 -1;

    float x = gl_VertexID - (gl_VertexID%2) - n;
    float y = float(n)*sign;

	// coordinates
    vec4 coordinate;
    if(dir == 0)
	    coordinate = vec4(x*dim, y*dim, 0, 1.0);
	else
        coordinate = vec4(y*dim, x*dim, 0, 1.0);

	// write point
	gl_Position = modelview * coordinate;
    
    if( (gl_VertexID / 2) %5 == 0 )
        objectColor = vec4(0.8, 0.8, 0.8, 0.5);
    else
        objectColor = vec4(0.8, 0.8, 0.8, 0.2);
}

    