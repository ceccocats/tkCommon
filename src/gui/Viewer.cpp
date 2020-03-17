#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/OBJ_Loader.h"


extern bool gRun;
using namespace tk::gui;
using namespace rl;

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
    camera.position = (Vector3){ 5.0f, 5.0f, 5.0f };    // Camera position
    camera.target = (Vector3){ 0.0f, 0.0f, 0.0f };      // Camera looking at point
    camera.up = (Vector3){ 0.0f, 0.0f, 1.0f };          // Camera up vector (rotation towards target)
    camera.fovy = 60.0f;                                // Camera field-of-view Y

    glfwGetFramebufferSize(window, &width, &height);
    clsSuc("init with resolution " + std::to_string(width) + "x" + std::to_string(height) + "\n");
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
    if(alpha >= 0)
        rlColor4ub(c.r, c.g, c.b, alpha*255);
    else
        rlColor4ub(c.r, c.g, c.b, c.a);
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









