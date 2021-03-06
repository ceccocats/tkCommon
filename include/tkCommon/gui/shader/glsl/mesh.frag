#version 330 core
out vec4 FragColor;

uniform vec4 color;
uniform vec3 lightPos;
uniform float ambientStrength;
uniform bool useLight;

in vec3 Normal;  
in vec3 FragPos;    

//light params
const vec3  lightColor          = vec3(1.0, 1.0, 1.0);

void main()
{
    if(useLight == false){
        FragColor = color;
    }else{
        // ambient
        vec3 ambient = ambientStrength * lightColor;
        
        // diffuse 
        vec3 norm       = normalize(Normal);
        vec3 lightDir   = normalize(lightPos - FragPos);
        float diff      = max(dot(norm, lightDir), 0.0);
        vec3 diffuse    = diff * lightColor;

        vec3 result = (ambient + diffuse) * color.xyz;
        FragColor   = vec4(result, color.w);
    }
} 