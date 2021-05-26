#version 330 core

layout (location = 0) in vec4  point;
layout (location = 1) in float r_feature;
layout (location = 2) in float g_feature;
layout (location = 3) in float b_feature;

uniform mat4  modelview;
uniform float alpha;
uniform bool  red;
uniform bool  green;
uniform bool  blue;

uniform float r_min, r_max;
uniform float g_min, g_max;
uniform float b_min, b_max;

out vec4 color;

void main(){

	float r = 0.0f;
	if(red == true)
		r = (r_feature - r_min) / (r_max - r_min); 

	float g = 0.0f;
	if(green == true)
		g = (g_feature - g_min) / (g_max - g_min);

	float b = 0.0f;
	if(blue == true)
		b = (b_feature - b_min) / (b_max - b_min);

	color 	= vec4(r,g,b,alpha);
	gl_Position = modelview * vec4(point.xyz, 1.0);
}
