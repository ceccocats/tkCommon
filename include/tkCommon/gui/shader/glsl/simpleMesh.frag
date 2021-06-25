#version 330 core

out vec4 	FragColor;

in vec3 	Normal;  
in vec3 	FragPos; 

uniform vec4 	color;

//light params
uniform float 	ambientStrength;
uniform bool 	useLight;
uniform vec3	lightPos;
const vec3  lightColor	= vec3(1.0, 1.0, 1.0);

void main() {
	if (useLight == true) { 
		// ambient
        vec3 ambient = ambientStrength * lightColor;
        
        // diffuse 
        vec3 norm       = normalize(Normal);
        vec3 lightDir   = normalize(lightPos - FragPos);
        float diff      = max(dot(norm, lightDir), 0.0);
        vec3 diffuse    = diff * lightColor;

        vec3 result = (ambient + diffuse) * color.xyz;
        FragColor   = vec4(result, color.w);
		return;
	} else {
		FragColor = color;
		return;
	}
}