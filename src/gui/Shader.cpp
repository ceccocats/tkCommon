#include "tkCommon/gui/Shader.h"

bool 
tk::gui::Shader::init(std::string vertexPath, std::string fragmentPath, std::string geometryPath){
    std::string  vertexCode, fragmentCode, geometryCode;
    unsigned int vertex, fragment, geometry;
    
    vertexCode = load(vertexPath);
    if(vertexCode == ""){
        return false;
    }

    fragmentCode = load(fragmentPath);
    if(fragmentCode == ""){
        return false;
    }

    if(!geometryPath.empty())
    {
        geometryCode = load(geometryPath);
        if(geometryCode == ""){
            return false;
        }
    }
    const char* vShaderCode = vertexCode.c_str();
    const char* fShaderCode = fragmentCode.c_str();
    const char* gShaderCode = geometryCode.c_str();

    vertex     = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vertex, 1, &vShaderCode, NULL);
    glCompileShader(vertex);
    checkCompileErrors(vertex, "VERTEX", vertexPath);


    fragment   = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fragment, 1, &fShaderCode, NULL);
    glCompileShader(fragment);
    checkCompileErrors(fragment, "FRAGMENT", fragmentPath);

    if(!geometryPath.empty())
    {
        geometry = glCreateShader(GL_GEOMETRY_SHADER);
        glShaderSource(geometry, 1, &gShaderCode, NULL);
        glCompileShader(geometry);
        checkCompileErrors(geometry, "GEOMETRY", geometryPath);
    }


    ID = glCreateProgram();
    glAttachShader(ID, vertex);
    glAttachShader(ID, fragment);
    if(!geometryPath.empty()){
        glAttachShader(ID, geometry);
    }

    glLinkProgram(ID);
    checkCompileErrors(ID, "PROGRAM", "Linking error");
    
    glDeleteShader(vertex);
    glDeleteShader(fragment);
    if(!geometryPath.empty()){
        glDeleteShader(geometry);
    }
    
    return true;
}

void 
tk::gui::Shader::use() { 
    glUseProgram(ID); 
}

void 
tk::gui::Shader::unuse() { 
    glUseProgram(0); 
}

bool 
tk::gui::Shader::close() { 
    glDeleteShader(ID);
    return true; 
}

void 
tk::gui::Shader::setBool(const std::string &name, bool value){         
    glUniform1i(glGetUniformLocation(ID, name.c_str()), (int)value); 
}

void 
tk::gui::Shader::setInt(const std::string &name, int value){ 
    glUniform1i(glGetUniformLocation(ID, name.c_str()), value); 
}

void 
tk::gui::Shader::setFloat(const std::string &name, float value){ 
    glUniform1f(glGetUniformLocation(ID, name.c_str()), value); 
}

void 
tk::gui::Shader::setVec2(const std::string &name, const glm::vec2 &value){ 
    glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]); 
}
void 
tk::gui::Shader::setVec2(const std::string &name, float x, float y){ 
    glUniform2f(glGetUniformLocation(ID, name.c_str()), x, y); 
}

void 
tk::gui::Shader::setVec3(const std::string &name, const glm::vec3 &value){ 
    glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]); 
}

void 
tk::gui::Shader::setVec3(const std::string &name, float x, float y, float z){ 
    glUniform3f(glGetUniformLocation(ID, name.c_str()), x, y, z); 
}

void 
tk::gui::Shader::setVec4(const std::string &name, const glm::vec4 &value){ 
    glUniform4fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]); 
}
void 
tk::gui::Shader::setVec4(const std::string &name, float x, float y, float z, float w){ 
    glUniform4f(glGetUniformLocation(ID, name.c_str()), x, y, z, w); 
}

void 
tk::gui::Shader::setMat2(const std::string &name, const glm::mat2 &mat){
    glUniformMatrix2fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void 
tk::gui::Shader::setMat3(const std::string &name, const glm::mat3 &mat){
    glUniformMatrix3fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void 
tk::gui::Shader::setMat4(const std::string &name, const glm::mat4 &mat){
    glUniformMatrix4fv(glGetUniformLocation(ID, name.c_str()), 1, GL_FALSE, &mat[0][0]);
}

void 
tk::gui::Shader::checkCompileErrors(GLuint shader, std::string type, std::string filename){
    GLint success;
    GLchar infoLog[1024];
    if(type != "PROGRAM")
    {
        glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
        if(!success)
        {
            glGetShaderInfoLog(shader, 1024, NULL, infoLog);
            clsErr(filename+"\n"+std::string{infoLog}+"\n");
        }
    }
    else
    {
        glGetProgramiv(shader, GL_LINK_STATUS, &success);
        if(!success)
        {
            glGetProgramInfoLog(shader, 1024, NULL, infoLog);
            clsErr(filename+"\n"+std::string{infoLog}+"\n");
        }
    }
}

std::string 
tk::gui::Shader::load(std::string path){
    std::string     includeIndentifier  = "#include ";
    std::string     fullSourceCode      = "";
    std::ifstream   file(path);
    std::string     lineBuffer;

    if (!file.is_open())
    {
        clsErr("Error opening file" + path + "\n");
        return "";
    }

    while (std::getline(file, lineBuffer))
    {
        if (lineBuffer.find(includeIndentifier) != lineBuffer.npos)
        {
            lineBuffer.erase(0, includeIndentifier.size());

            //removing "" from include
            lineBuffer = lineBuffer.substr(1,lineBuffer.size()-2);

            size_t found = path.find_last_of("/\\");
            lineBuffer.insert(0, path.substr(0, found + 1));

            fullSourceCode += load(lineBuffer);

            continue;
        }
        fullSourceCode += lineBuffer + '\n';
    }
    file.close();

    return fullSourceCode;
}