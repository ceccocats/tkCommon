#version 330 core
#include "../colormaps/MATLAB_hsv.frag"

out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  
  
uniform vec3 lightPos;  

const float range = 20.0f;

//light params
const vec3  lightColor          = vec3(1.0, 1.0, 1.0);
const float ambientStrength     = 0.5;
const float opacity             = 1.0;

void main()
{
	vec3 objectColor = colormap(mod(FragPos.z,range)/range).xyz;

    // ambient
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm       = normalize(Normal);
    vec3 lightDir   = normalize(lightPos - FragPos);
    float diff      = max(dot(norm, lightDir), 0.0);
    vec3 diffuse    = diff * lightColor;

    vec3 result = (ambient + diffuse) * objectColor;
    FragColor   = vec4(result, opacity);
} 