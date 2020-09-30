#include "tkCommon/gui/Viewer.h"

namespace tk { namespace gui {
Viewer* Viewer::instance = nullptr;

void Viewer::init() {
    clsMsg("init\n");

    glfwSetErrorCallback(errorCallback);
    glfwInit();

    //Anti-Aliasing
    glfwWindowHint(GLFW_SAMPLES, 3);

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(width, height, windowName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
    }

    //Icon
    int channels;
    GLFWimage img;
    img.pixels =tk::gui::common::loadImage(std::string(tkCommon_PATH) + "data/tkLogo.png", &img.width, &img.height, &channels);
    glfwSetWindowIcon(window,1,&img);  

    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, keyCallback);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glCheckError();

    // Initialize OpenGL loader
    bool err = glewInit() != GLEW_OK;
    if (err) {
        tk::tprint::printErr("Viewer", "Failed to initialize OpenGL loader!\n");
        exit(-1);
    }
    int foo = 1;
    const char* bar[1] = {" "}; 
    glutInit(&foo, (char**)bar);

    axis.init();
    grid.init();
    text.init(fontPath);

    glCheckError();

    //ImGUI
    {
        // Setup Dear ImGui context
        IMGUI_CHECKVERSION();
        ImGui::CreateContext();
        ImGuiIO& io = ImGui::GetIO(); (void)io;
        // Setup Dear ImGui style
        ImGui::StyleColorsDark();
        //ImGui::StyleColorsClassic();

        // Setup Platform/Renderer bindings
        ImGui_ImplGlfw_InitForOpenGL(window, true);
        ImGui_ImplOpenGL3_Init(glsl_version);
    }


    // OpenGL confs
    glEnable(GL_DEPTH_TEST);
    //glDisable(GL_CULL_FACE);        
    //glDepthFunc(GL_GEQUAL);
    //glEnable(GL_LINE_SMOOTH);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    //glEnable(GL_ALPHA_TEST);
    //glDepthMask(GL_FALSE); // make the depth buffer read-only
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    
    // print OpenGL status
    std::string msg = std::string{"OPENGL running on:"} + 
                        std::string(reinterpret_cast<const char*>(glGetString(GL_VERSION))) + " -- " +
                        std::string(reinterpret_cast<const char*>(glGetString(GL_RENDERER))) + "\n";
    clsMsg(msg);

    // set window size
    glfwGetFramebufferSize(window, &width, &height);
    clsSuc("init with resolution " + std::to_string(width) + "x" + std::to_string(height) + "\n");

    // init cameras
    camera.init();

    glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);

    // lights
    lightPos = glm::vec3(0.0f, 0.0f, 20.0f);

    glCheckError();
}

void Viewer::draw() {
    drawBuffer.draw(this);

    axis.draw();
    grid.draw();
    text.draw("gatti gay", 1200.0f, 25.0f);

    glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);

    ImGuiIO& io = ImGui::GetIO();
    ImGui::Begin("Viewer");
    ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
    ImGui::Text("window size: %d x %d", width, height);
    ImGui::Text("window ratio: %f", aspectRatio);
    ImGui::Text("GPU memory: %d / %d MB", (int)((total_mem_kb - cur_avail_mem_kb)/1024.0), (int)(total_mem_kb/1024.0));
    ImGui::Text("Drawables (%d)", int(drawBuffer.map.size()));
    std::map<std::string,Drawable*>::iterator it; 
    for(it = drawBuffer.map.begin(); it!=drawBuffer.map.end(); ++it){
        ImGui::Checkbox(it->first.c_str(), &it->second->enabled); 
        ImGui::SameLine(ImGui::GetWindowWidth()-30);
        ImGui::Checkbox( ("##" + it->first +"_follow").c_str(), &it->second->follow); 
    }
    ImGui::End();

    glCheckError();
}

void Viewer::run() {
    clsMsg("run\n");
    running = true;
    timeStamp_t VIZ_DT_US = dt*1e6;
    LoopRate rate((VIZ_DT_US), "VIZ_UPDATE");
    while (running) {

        // manage window resize
        glfwGetFramebufferSize(window, &width, &height);
        aspectRatio = (float)width / (float)height;
        xLim = aspectRatio;
        yLim = 1.0;

        // average centers of every drawable to follow
        if(drawBuffer.centers.size() >0) {
            tk::common::Vector3<float> center;
            for(int i=0; i<drawBuffer.centers.size(); i++) {
                center.x += drawBuffer.centers[i].x;
                center.y += drawBuffer.centers[i].y;
                center.z += drawBuffer.centers[i].z;
            }
            center.x /= drawBuffer.centers.size();
            center.y /= drawBuffer.centers.size();
            center.z /= drawBuffer.centers.size();
            camera.setCenter(center);
        }
        
        camera.setViewPort(0, 0, width, height);
        glViewport(camera.viewport[0], camera.viewport[1], 
                    camera.viewport[2], camera.viewport[3]);

        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity() ;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(float(background.r)/255, float(background.g)/255, float(background.b)/255, float(background.a)/255);

        glPushMatrix();

        // apply camera matrix
        glMultMatrixf(glm::value_ptr(camera.projection));
        glMultMatrixf(glm::value_ptr(camera.modelView));
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        camera.mouseOnGUI = ImGui::IsMouseHoveringAnyWindow();

        // just for debug
        glm::vec3 pp = camera.unprojectPlane({camera.mousePos.x,camera.mousePos.y});
        // pp is mouse pose

        draw();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glPopMatrix();

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
        // update running status
        running = running && !glfwWindowShouldClose(window);
        rate.wait(false);
    }

    drawBuffer.close();

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();  
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();
}

bool Viewer::isRunning() {
    return running;
}

void Viewer::close() { 
    axis.close();
    grid.close();
    text.close();
    clsMsg("close\n");
    running = false;
}

void Viewer::add(std::string name, Drawable *data){
    drawBuffer.add(name, data, this);
}

void Viewer::errorCallback(int error, const char* description) {
    tk::tprint::printErr("Viewer", std::string{"error: "}+std::to_string(error)+" "+description+"\n");
}

void Viewer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    Viewer::instance->camera.mouseWheel(xoffset, yoffset);
}

void Viewer::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    Viewer::instance->camera.mouseMove(xpos, ypos);
}

void Viewer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
        Viewer::instance->camera.mouseDown(button, xpos, ypos);
    if (action == GLFW_RELEASE)
        Viewer::instance->camera.mouseUp(button, xpos, ypos);
}

void Viewer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){

    if (key == GLFW_KEY_ESCAPE){
        glfwSetWindowShouldClose(window, true); 	
    }

    if(key == GLFW_KEY_R){
        Viewer::instance->camera.init();
    }
}

}}

