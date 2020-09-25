#version 330 core
out vec4 FragColor;

in vec3 Normal;  
in vec3 FragPos;  
in vec4 objectColor;
  
uniform vec3 lightPos;  

//light params
const vec3  lightColor          = vec3(1.0, 1.0, 1.0);
const float ambientStrength     = 0.6;

void main()
{
    // ambient
    vec3 ambient = ambientStrength * lightColor;
  	
    // diffuse 
    vec3 norm       = normalize(Normal);
    vec3 lightDir   = normalize(lightPos - FragPos);
    float diff      = max(dot(norm, lightDir), 0.0);
    vec3 diffuse    = diff * lightColor;

    vec3 result = (ambient + diffuse) * objectColor.xyz;
    FragColor   = vec4(result, objectColor.w);
} 

////////////////////////////////////////////////////////////
/*  transparent rendering
    https://www.alecjacobson.com/weblog/?p=2750
*/

/*  specular

    uniform vec3 viewPos;

    // specular
    const float specularStrength    = 0.0;
    vec3 viewDir    = normalize(viewPos - FragPos);
    vec3 reflectDir = reflect(-lightDir, norm);  
    float spec      = pow(max(dot(viewDir, reflectDir), 0.0), 32);
    vec3 specular   = specularStrength * spec * lightColor;  
        
    vec3 result = (ambient + diffuse + specular) * objectColor;
*/