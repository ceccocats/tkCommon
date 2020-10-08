#include "tkCommon/gui/Viewer.h"

namespace tk { namespace gui {
Viewer* Viewer::instance = nullptr;

Viewer::Viewer(){
    Viewer::instance = this;
}


Viewer::~Viewer(){

}

void 
Viewer::start(){
    if(running == false){
        glThread.init(run,Viewer::instance);
    }else{
        clsWrn("Thread is already started\n");
    }
}

bool 
Viewer::isRunning(){
    return running;
}

void 
Viewer::stop(){
    running = false;
}

void 
Viewer::join(){
    glThread.join();
}

void 
Viewer::add(tk::gui::Drawable* obj){
    newDrawables[newDrawables.size()] = obj;
}

void* 
Viewer::run(void* istance){
    Viewer* viewer = (Viewer*)istance;

    viewer->init();

    viewer->runloop();
}

void 
Viewer::init() {
    glfwSetErrorCallback(errorCallback);
    glfwInit();

    //Anti-Aliasing
    glfwWindowHint(GLFW_SAMPLES, 3);

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(width, height, windowName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
    }

    //glut init
    int argcp = 0;
    char** argv;
    glutInit(&argcp, argv);

    //Icon
    int channels;
    GLFWimage img;
    img.pixels =tk::gui::common::loadImage(std::string(tkCommon_PATH) + "data/tkLogo.png", &img.width, &img.height, &channels);
    glfwSetWindowIcon(window,1,&img); 
    glCheckError(); 

    //Callbacks
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, keyCallback);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);
    glCheckError();


    // Initialize OpenGL loader
    if (glewInit() != GLEW_OK) {
        tk::tprint::printErr("Viewer", "Failed to initialize OpenGL loader!\n");
        exit(-1);
    }

    //ImGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // OpenGL confs
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);

    // set window size
    glfwGetFramebufferSize(window, &width, &height);

    // init cameras
    camera.init();

    glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);

    // lights
    lightPos = glm::vec3(0.0f, 0.0f, 20.0f);

    clsSuc  (   std::string{"OPENGL running on: "} + 
                std::string(reinterpret_cast<const char*>(glGetString(GL_VERSION))) + " -- " +
                std::string(reinterpret_cast<const char*>(glGetString(GL_RENDERER))) + "\n"
            );

    glCheckError();
}

void 
Viewer::imguiDraw(){
    ImGui::SetNextWindowSize(ImVec2(500, 440), ImGuiCond_FirstUseEver);
    ImGui::Begin("tkGUI");

    ImGui::BeginChild("left pane", ImVec2(150, 0), true);
    if(ImGui::Selectable("Viewer", imguiSelected == -1))
            imguiSelected = -1;
    ImGui::Separator();
    for(int i = 0; i < drawables.size(); i++){
        std::string selectName = drawables[i]->toString();

        if(drawables[i]->enabled == false){
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
            drawables[i]->follow = false;
        }
        
        if(drawables[i]->follow == true)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 1.0f, 0.5f, 1.0f));
    
        if(ImGui::Selectable(selectName.c_str(), imguiSelected == i))
            imguiSelected = i;
        
        if(drawables[i]->enabled == false || drawables[i]->follow == true)
            ImGui::PopStyleColor();

    }
    ImGui::EndChild();
    ImGui::SameLine();

    ImGui::BeginGroup();
    ImGui::BeginChild("item view", ImVec2(0, -ImGui::GetFrameHeightWithSpacing()));
    std::string name;
    if(imguiSelected == -1){
        name = "viewer";
    }else{
        name = drawables[imguiSelected]->toString();
    }
    ImGui::Text("%s",name.c_str());
    ImGui::Separator();

    ImGui::BeginTabBar("##Tabs", ImGuiTabBarFlags_None);

    if(ImGui::BeginTabItem("Info")){
        drawInfos();
        ImGui::EndTabItem();
    }

    if(ImGui::BeginTabItem("Settings")){
        drawSettings();
        ImGui::EndTabItem();
    }

    ImGui::EndTabBar();
    ImGui::EndChild();

    if(imguiSelected != -1){

        std::string value;
        if(drawables[imguiSelected]->follow == true){
            value = "unfollow";
        }else{
            value = "follow";
        }
        if (ImGui::Button(value.c_str())){
            drawables[imguiSelected]->follow = ! drawables[imguiSelected]->follow;
        }
        ImGui::SameLine();

        if(drawables[imguiSelected]->enabled == true){
            value = "disable";
        }else{
            value = "enable";
        }
        if (ImGui::Button(value.c_str())){
            drawables[imguiSelected]->enabled = ! drawables[imguiSelected]->enabled;
        }
    }
    ImGui::EndGroup();
    ImGui::End();

}

void 
Viewer::drawInfos(){

    if(imguiSelected == -1){
        ImGuiIO& io = ImGui::GetIO();
        ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / io.Framerate, io.Framerate);
        ImGui::Text("Window size: %d x %d", width, height);
        ImGui::Text("window ratio: %f", aspectRatio);
        glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);
        ImGui::Text("GPU memory: %d / %d MB", (int)((total_mem_kb - cur_avail_mem_kb)/1024.0), (int)(total_mem_kb/1024.0));
    }else{
        drawables[imguiSelected]->imGuiInfos();
    }
    glCheckError();
}

void 
Viewer::drawSettings(){
    if(imguiSelected == -1){
        ImGui::Text("La mamma di gatti");
    }else{
        drawables[imguiSelected]->imGuiSettings();
    }
    glCheckError();
}

void
Viewer::beforeDraw(){
    for (auto const& drawable : drawables){
        if(drawable.second->enabled) //TODO: removing?!?
            drawable.second->beforeDraw(Viewer::instance);
    }
    glCheckError();
}

void 
Viewer::draw() {
    for (auto const& drawable : drawables){
        if(drawable.second->enabled){
            glPushMatrix();
			glMultMatrixf(drawable.second->tf.matrix().data());
            drawable.second->draw(Viewer::instance);
            glPopMatrix();
        }
    }
    glCheckError();
}

void 
Viewer::draw2D() {

    //TODO: mettere la camera in ambiente 2D e disegnare logo tk di default

    for (auto const& drawable : drawables){
        if(drawable.second->enabled)
            drawable.second->draw2D(Viewer::instance);
    }
    glCheckError();
}

void 
Viewer::initDrawables() {

    if(newDrawables.size() > 0){
        for (auto const& drw : newDrawables){
            drw.second->onInit(Viewer::instance);
            drawables[drawables.size()] = drw.second;
        }
        newDrawables.clear();
    }
    glCheckError();
}

void 
Viewer::follow() {

    tk::common::Vector3<float> center;
    int n = 0;
    for (auto const& drawable : drawables){
        if(drawable.second->follow){
            center += tk::common::tf2pose(drawable.second->tf);
            n++;
        }
    }

    if(n > 0){
        center.x /= n;
        center.y /= n;
        center.z /= n;
        camera.setCenter(center);
    }
}

void 
Viewer::close() { 
    stop();
    join();
    for (auto const& drawable : drawables){
        drawable.second->onClose();
        delete drawable.second;
    }   
    clsMsg("closed\n");
}

void Viewer::runloop() {

    running = true;
    timeStamp_t VIZ_DT_US = dt*1e6;
    LoopRate rate((VIZ_DT_US), "VIZ_UPDATE");

    while (running) {

        // manage window resize
        glfwGetFramebufferSize(window, &width, &height);
        aspectRatio = (float)width / (float)height;
        xLim = aspectRatio;
        yLim = 1.0;

        initDrawables();
        follow();
        
        camera.setViewPort(0, 0, width, height);
        glViewport(camera.viewport[0], camera.viewport[1], camera.viewport[2], camera.viewport[3]);

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

        // just for debug -- pp is mouse pose
        glm::vec3 pp = camera.unprojectPlane({camera.mousePos.x,camera.mousePos.y});

        //tkRendering
        imguiDraw();
        draw();
        draw2D();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        glPopMatrix();

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
        glCheckError();
        running = running && !glfwWindowShouldClose(window);
        rate.wait(false);
    }

    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();  
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();

    close();
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

