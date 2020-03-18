#define RLGL_IMPLEMENTATION
#define RLGL_STANDALONE
#define GRAPHICS_API_OPENGL_33
#include "tkCommon/gui/raylib/rlgl.h" 
#include "tkCommon/gui/raylib/rlgl_data.h"
#include "tkCommon/gui/raylib/rlgl_utils.h"

#include "tkCommon/gui/Viewer.h"

extern bool gRun;
using namespace tk::gui;
using namespace rl;

rl::Color tkCurrentColor;

Viewer::Viewer() {
    // this must stay in the constructor because the file is loaded in the init function
    icon_fname = std::string(std::string(TKPROJ_PATH)+"/data/tkLogo.png");
}


Viewer::~Viewer() {
}

void 
Viewer::init() {
    glfwSetErrorCallback(errorCallback);
    glfwInit();

    GLFWimage icons[1];

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(width, height, windowName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
    }


#if GLFW_VERSION_MAJOR >= 3
#if GLFW_VERSION_MINOR >= 2
    //glfwMaximizeWindow(window);

    unsigned w, h;
    unsigned err = lodepng_decode32_file(&(icons[0].pixels), &w, &h, icon_fname.c_str());
    icons[0].width = w;
    icons[0].height = h;
    clsMsg("loading icon: " + icon_fname + ": " + std::to_string(err) + "\n");

    glfwSetWindowIcon(window, 1, icons);
#endif
#endif


    glfwSetScrollCallback(window, Viewer::scroll_callback);
    glfwSetCursorPosCallback(window, Viewer::cursor_position_callback);
    glfwSetMouseButtonCallback(window, Viewer::mouse_button_callback);

    glfwSetKeyCallback(window, keyCallback);
    glfwMakeContextCurrent(window);
    //rladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);

    // RLGL init
    rlLoadExtensions((void*)glfwGetProcAddress);
    rlglInit(width, height);

    // Initialize viewport and internal projection/modelview matrices
    rlViewport(0, 0, width, height);
    rlMatrixMode(RL_PROJECTION);                        // Switch to PROJECTION matrix
    rlLoadIdentity();                                   // Reset current matrix (PROJECTION)
    rlOrtho(0, width, height, 0, 0.0f, 1.0f);   // Orthographic projection with top-left corner at (0,0)
    rlMatrixMode(RL_MODELVIEW);                         // Switch back to MODELVIEW matrix
    rlLoadIdentity();                                   // Reset current matrix (MODELVIEW)

    rlClearColor(float(background.r), float(background.g), float(background.b), float(background.a));
    rlEnableDepthTest();                                // Enable DEPTH_TEST for 3D

    camera = { 0 };
    camera.position = (Vector3){ 15.0f, 15.0f, 15.0f };    // Camera position
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at point
    camera.up = (Vector3){ 0.0f, 0.0f, 1.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 60.0f;                                // Camera field-of-view Y

    glfwGetFramebufferSize(window, &width, &height);
    clsSuc("init with resolution " + std::to_string(width) + "x" + std::to_string(height) + "\n");

    tkGenCubeModel(primitives[TK_CUBE]);
    tkGenCircleModel(primitives[TK_CIRCLE]);
    tkGenSphereModel(primitives[TK_SPHERE]);

    tkSetColor(tk::gui::color::WHITE);
}


void 
Viewer::draw() {
    tkDrawAxis();
}

void
Viewer::draw2d() {

}

void
Viewer::drawSplash() {

}

void 
Viewer::run() {
    timeStamp_t VIZ_DT_US = dt*1e6;
    LoopRate rate((VIZ_DT_US), "VIZ_UPDATE");
    while (gRun && !glfwWindowShouldClose(window)) {

        glfwGetFramebufferSize(window, &width, &height);
        aspectRatio = (float)width / (float)height;
        xLim = aspectRatio;
        yLim = 1.0;


        rlClearScreenBuffers();

        // 3d draw 
        Matrix matProj = MatrixPerspective(camera.fovy*DEG2RAD, (double)width/(double)height, 0.01, 1000.0);
        Matrix matView = MatrixLookAt(camera.position, camera.target, camera.up);
        SetMatrixModelview(matView);    // Set internal modelview matrix (default shader)
        SetMatrixProjection(matProj);   // Set internal projection matrix (default shader) 

        draw();

        rlglDraw();

        // 2D draw
        rlMatrixMode(RL_PROJECTION);                            // Enable internal projection matrix
        rlLoadIdentity();                                       // Reset internal projection matrix
        rlOrtho(0.0, width, height, 0.0, 0.0, 1.0); // Recalculate internal projection matrix
        rlMatrixMode(RL_MODELVIEW);                             // Enable internal modelview matrix
        rlLoadIdentity();                                       // Reset internal modelview matrix

        draw2d();

        rlglDraw();

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
        rate.wait(false);
    }
    gRun = false;

    glfwDestroyWindow(window);
    glfwTerminate();

    rlglClose();                    // Unload rlgl internal buffers and default shader/texture
}

void
Viewer::setIcon(std::string filename) {
    icon_fname = filename;
#if GLFW_VERSION_MINOR < 2
    clsWrn("To load program icon update GLFW version to 3.3")
#endif
}

void
Viewer::setWindowName(std::string name) {
    windowName = name;
}


void 
Viewer::setBackground(tk::gui::Color_t c) {
    background = c;
}

void 
Viewer::tkSetColor(tk::gui::Color_t c, float alpha) {
    if(alpha >= 0) {
        rlColor4ub(c.r, c.g, c.b, alpha*255);
        tkCurrentColor = { c.r, c.g, c.b, alpha*255 };
    } else {
        rlColor4ub(c.r, c.g, c.b, c.a);
        tkCurrentColor = { c.r, c.g, c.b, c.a };
    }
}

void 
Viewer::tkDrawAxis(float s) {

    //rlLineWidth(2.5); 

    rlColor4ub(255, 0, 0, 255);
    rlBegin(RL_LINES);
    // x
    rlVertex3f(0, 0, 0);
    rlVertex3f(s, 0, 0);
    rlEnd();

    rlColor4ub(0, 255, 0, 255);
    rlBegin(RL_LINES);
    // y
    rlVertex3f(0, 0, 0);
    rlVertex3f(0, s, 0);
    rlEnd();

    rlColor4ub(0, 0, 255, 255);
    rlBegin(RL_LINES);
    // z
    rlVertex3f(0, 0, 0);
    rlVertex3f(0, 0, s);
    rlEnd();
}


void 
Viewer::errorCallback(int error, const char* description) {
    tk::tformat::printErr("Viewer", std::string{"error: "}+std::to_string(error)+" "+description+"\n");
}

void
Viewer::tkGenModel(float *vertices, int n, rl::Model &obj) {
    const int MAX_MESH_VBO = 7;
    Mesh mesh = {0};
    mesh.vboId = (unsigned int *)RL_CALLOC(MAX_MESH_VBO, sizeof(unsigned int));
    
    mesh.vertices = (float *)RL_MALLOC(n*3*sizeof(float));
    memcpy(mesh.vertices, vertices, n*3*sizeof(float));

    mesh.vertexCount = n;
    mesh.triangleCount = n/3;

    rl::rlLoadMesh(&mesh, false);
    obj = rl::LoadModelFromMesh(mesh);
}

void
Viewer::tkGenCubeModel(rl::Model &obj) {
    const int n = 3*12;  
    float vertices[n*3] = {
        -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, -0.5, +0.5, +0.5,
        +0.5, +0.5, +0.5, -0.5, +0.5, +0.5, +0.5, -0.5, +0.5,
        -0.5, -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5,
        +0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5, +0.5, -0.5,
        -0.5, +0.5, -0.5, -0.5, +0.5, +0.5, +0.5, +0.5, +0.5,
        +0.5, +0.5, -0.5, -0.5, +0.5, -0.5, +0.5, +0.5, +0.5,
        -0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, +0.5,
        +0.5, -0.5, -0.5, +0.5, -0.5, +0.5, -0.5, -0.5, -0.5,
        +0.5, -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, +0.5, +0.5,
        +0.5, -0.5, +0.5, +0.5, -0.5, -0.5, +0.5, +0.5, +0.5,
        -0.5, -0.5, -0.5, -0.5, +0.5, +0.5, -0.5, +0.5, -0.5,
        -0.5, -0.5, +0.5, -0.5, +0.5, +0.5, -0.5, -0.5, -0.5
    };
    tkGenModel(vertices, n, obj);
}

void
Viewer::tkGenCircleModel(rl::Model &obj, int res) {

    float vertices[res*9];
    for (int j = 0; j < res; j++)   {
        float theta = 2.0f * 3.1415926f * float(j) / float(res);//get the current angle 
        float xr0 = 1.0 * cosf(theta);//calculate the x component 
        float yr0 = 1.0 * sinf(theta);//calculate the y component
        
        theta = 2.0f * 3.1415926f * float(j+1%res) / float(res);//get the current angle 
        float xr1 = 1.0 * cosf(theta);//calculate the x component 
        float yr1 = 1.0 * sinf(theta);//calculate the y component
        
        vertices[j*9+0] = xr0;
        vertices[j*9+1] = yr0;
        vertices[j*9+2] = 0;
        vertices[j*9+3] = xr1;
        vertices[j*9+4] = yr1;
        vertices[j*9+5] = 0;
        vertices[j*9+6] = 0;
        vertices[j*9+7] = 0;
        vertices[j*9+8] = 0;
    }
    tkGenModel(vertices, res*3, obj);
}

void
Viewer::tkGenSphereModel(rl::Model &obj, int res) {

    std::vector<float> vertices;
    int rings = res;
    int slices = res;

    for (int i = 0; i < (rings +2); i++) {
        for (int j = 0; j < slices; j++) {
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*i))*sinf(DEG2RAD*(j*360/slices)));
            vertices.push_back(sinf(DEG2RAD*(270+(180/(rings + 1))*i)));
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*i))*cosf(DEG2RAD*(j*360/slices)));
            
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*(i+1)))*sinf(DEG2RAD*((j+1)*360/slices)));
            vertices.push_back(sinf(DEG2RAD*(270+(180/(rings + 1))*(i+1))));
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*(i+1)))*cosf(DEG2RAD*((j+1)*360/slices)));
            
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*(i+1)))*sinf(DEG2RAD*(j*360/slices)));
            vertices.push_back(sinf(DEG2RAD*(270+(180/(rings + 1))*(i+1))));
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*(i+1)))*cosf(DEG2RAD*(j*360/slices)));
            
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*i))*sinf(DEG2RAD*(j*360/slices)));
            vertices.push_back(sinf(DEG2RAD*(270+(180/(rings + 1))*i)));
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*i))*cosf(DEG2RAD*(j*360/slices)));
            
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*(i)))*sinf(DEG2RAD*((j+1)*360/slices)));
            vertices.push_back(sinf(DEG2RAD*(270+(180/(rings + 1))*(i))));
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*(i)))*cosf(DEG2RAD*((j+1)*360/slices)));
            
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*(i+1)))*sinf(DEG2RAD*((j+1)*360/slices)));
            vertices.push_back(sinf(DEG2RAD*(270+(180/(rings + 1))*(i+1))));
            vertices.push_back(cosf(DEG2RAD*(270+(180/(rings + 1))*(i+1)))*cosf(DEG2RAD*((j+1)*360/slices)));
        }
    }

    tkGenModel(vertices.data(), vertices.size()/3, obj);
}

void
Viewer::tkApplyManip(Manip_t manip) {
    rlPushMatrix();

    rlTranslatef(manip.pose.x, manip.pose.y, manip.pose.z);
    rlRotatef(RAD2DEG*manip.rot.x, 1, 0, 0);
    rlRotatef(RAD2DEG*manip.rot.y, 0, 1, 0);
    rlRotatef(RAD2DEG*manip.rot.z, 0, 0, 1);
    rlScalef(manip.scale.x, manip.scale.y, manip.scale.z);
}

void
Viewer::tkApplyTf(tk::common::Tfpose tf) {
    rlPushMatrix();
    rlMultMatrixf(tf.matrix().data());
}

void
Viewer::tkDeApply() {
    rlPopMatrix();
}

void 
Viewer::tkDrawModel(rl::Model &obj, Manip_t manip) {

    tkApplyManip(manip);

    rl::Color tint = tkCurrentColor;

    for(int i=0; i<obj.meshCount && i<obj.materialCount; i++) {
        
        rl::Color color = obj.materials[obj.meshMaterial[i]].maps[MAP_DIFFUSE].color;

        rl::Color colorTint = {255,255,255,255};
        colorTint.r = (((float)color.r/255.0)*((float)tint.r/255.0))*255;
        colorTint.g = (((float)color.g/255.0)*((float)tint.g/255.0))*255;
        colorTint.b = (((float)color.b/255.0)*((float)tint.b/255.0))*255;
        colorTint.a = (((float)color.a/255.0)*((float)tint.a/255.0))*255;

        obj.materials[obj.meshMaterial[i]].maps[MAP_DIFFUSE].color = colorTint;
        rl::rlDrawMesh(obj.meshes[i], obj.materials[i], obj.transform);

        // restore
        obj.materials[obj.meshMaterial[i]].maps[MAP_DIFFUSE].color = color;
    }

    tkDeApply();
}
