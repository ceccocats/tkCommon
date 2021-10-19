#include "tkCommon/gui/Viewer.h"
#include "tkCommon/rt/Task.h"
#include "tkCommon/rt/Profiler.h"

namespace tk { namespace gui {
Viewer* Viewer::instance = nullptr;
bool Viewer::disabled = false;

Viewer::Viewer(){

}


Viewer::~Viewer(){
    delete Viewer::instance;
}

void 
Viewer::start(bool useImGUI){
    if(Viewer::disabled) {
        running = true;
        glThread.init(fake_run,Viewer::instance);
        return;
    }

    if(running == false){
        running = true;
        this->useImGUI=useImGUI;
        glThread.init(run,Viewer::instance);
    }else{
        tkWRN("Thread is already started\n");
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
    pthread_exit(NULL);
}

void* 
Viewer::fake_run(void* istance){
    Viewer* viewer = (Viewer*)istance;
    while (viewer->isRunning()) {
        sleep(100);
    }
    pthread_exit(NULL);
}

float 
Viewer::getWidth(){
    return width;
}

float 
Viewer::getHeight(){
    return height;
}

glm::vec3 
Viewer::getLightPos(){
    return lightPos;
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
    img.pixels = tk::gui::common::loadImage(std::string(tkCommon_PATH) + "data/tkLogo.png", &img.width, &img.height, &channels);
    glfwSetWindowIcon(window,1,&img); 
    free(img.pixels);

    //Callbacks
    glfwSetScrollCallback(window, scroll_callback);
    glfwSetCursorPosCallback(window, cursor_position_callback);
    glfwSetMouseButtonCallback(window, mouse_button_callback);
    glfwSetKeyCallback(window, keyCallback);
    glfwSetWindowSizeCallback(window, window_size_callback);
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);


    // Initialize OpenGL loader
    if (glewInit() != GLEW_OK) {
        tkERR("Failed to initialize OpenGL loader!\n");
        exit(-1);
    }

    //ImGUI
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlot::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImPlot::GetStyle().AntiAliasedLines = true;

    // OpenGL confs
    glEnable(GL_MULTISAMPLE);
    glEnable(GL_DEPTH_TEST);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_BLEND);
    glEnable(GL_VERTEX_PROGRAM_POINT_SIZE);
    // culling 
    glEnable(GL_CULL_FACE);
    glCullFace(GL_BACK);
    glFrontFace(GL_CCW);

    // set window size
    glfwGetFramebufferSize(window, &width, &height);

    // init cameras
    camera.init();
    camera.setViewPort(0, 0, width, height);

    //Check that we are running on GPU not integrated
    std::string vendor = (const char*)glGetString(GL_VENDOR);
    std::string name   = (const char*)glGetString(GL_RENDERER);
    if(vendor.find("NVIDIA") != std::string::npos && name.find("integrated") == std::string::npos){
        gpu = true;
    }

    //Graphics
    for(int i = 0; i < nFPS; i++){
        vizFPS[i] = 0;
    }
    if(gpu){
        glGetIntegerv(GL_GPU_MEM_INFO_TOTAL_AVAILABLE_MEM_NVX, &total_mem_kb);
        for(int i = 0; i < nUsage; i++){
            gpuUsage[i] = 0;
        }
    }

    // lights
    lightPos = glm::vec3(0.0f, 0.0f, 100.0f);

    tkMSG(std::string{"OPENGL running on: "} + 
          std::string(reinterpret_cast<const char*>(glGetString(GL_VERSION))) + " -- " +
          std::string(reinterpret_cast<const char*>(glGetString(GL_RENDERER))) + "\n");

    //tkLogo
    int height, width;
    uint8_t* image = tk::gui::common::loadImage(std::string(tkCommon_PATH) + "data/er_tk.png", &width, &height, &channels);
    //Set trasparency
    for(int i = 0; i < width*height; i++){
        if(image[i*4] == 0)
            image[i*4 + 3] = 0;
        else{
            image[i*4] = 255;
            image[i*4 + 1] = 255;
            image[i*4 + 2] = 255;
            image[i*4 + 3] = 128;
        }
    }
    logo.init(width,height,channels);
    logo.setData(image);
    free(image);
    drwLogo = tk::gui::shader::texture::getInstance();
    pos.init();
}

void 
Viewer::imguiDraw(){
    ImGui::SetNextWindowSize(ImVec2(width, height/2), ImGuiCond_Always);
    ImGui::SetNextWindowPos(ImVec2(0, height/2), ImGuiCond_Always);
    ImGui::Begin("tkGUI");

    ImGui::BeginChild("left pane", ImVec2(150, 0), true);
    if(ImGui::Selectable("Viewer", imguiSelected == -1))
            imguiSelected = -1;
    ImGui::Separator();
    for(int i = 0; i < drawables.size(); i++){
        std::string selectName = drawables[i]->toString();

        if(selectName.empty())
            continue;

        if(drawables[i]->enabled == false){
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 1.0f, 1.0f, 0.4f));
            drawables[i]->follow = 0;
        }
        
        if(drawables[i]->follow == 1)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 1.0f, 0.5f, 1.0f));
        if(drawables[i]->follow == 2)
            ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(0.5f, 1.0f, 1.0f, 1.0f));
    

        if(ImGui::Selectable(selectName.c_str(), imguiSelected == i))
            imguiSelected = i;
        
        if(drawables[i]->enabled == false || drawables[i]->follow > 0)
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
        if(drawables[imguiSelected]->follow == 2){
            value = "Unfollow";
        } else if(drawables[imguiSelected]->follow == 1){
            value = "Follow Angle";
        } else {
            value = "Follow";
        }
        if (ImGui::Button(value.c_str())){
            drawables[imguiSelected]->follow = (drawables[imguiSelected]->follow+1) %3;
        }
        ImGui::SameLine();

        if(drawables[imguiSelected]->enabled == true){
            value = "Disable";
        }else{
            value = "Enable";
        }
        if (ImGui::Button(value.c_str())){
            drawables[imguiSelected]->enabled = ! drawables[imguiSelected]->enabled;
        }
        ImGui::SameLine();

        if(drawables[imguiSelected]->enableTf == true){
            value = "Disable tf";
        }else{
            value = "Draw tf";
        }
        if (ImGui::Button(value.c_str())){
            drawables[imguiSelected]->enableTf = ! drawables[imguiSelected]->enableTf;
        }
    }
    ImGui::EndGroup();
    ImGui::End();

}

void 
Viewer::drawInfos(){

    if(imguiSelected == -1){

        //Usage FPS and GPU memory
        static int update = 0;
        if(update++ % 30 == 0){
            if(gpu){
                glGetIntegerv(GL_GPU_MEM_INFO_CURRENT_AVAILABLE_MEM_NVX, &cur_avail_mem_kb);
                for(int i = nUsage-1; i > 0; i--){
                    gpuUsage[i] = gpuUsage[i-1];
                }
                gpuUsage[0] = (total_mem_kb - cur_avail_mem_kb)/1024.0;
            }

            for(int i = nFPS-1; i > 0; i--){
                vizFPS[i] = vizFPS[i-1];
            }
            ImGuiIO& io = ImGui::GetIO();
            vizFPS[0] = io.Framerate;
        }
        
        if(gpu){
            std::string text = "GPU memory";
            std::string usage = std::to_string((int)((total_mem_kb - cur_avail_mem_kb)/1024.0)) + " / " + std::to_string((int)(total_mem_kb/1024.0)) + " MB";
            ImGui::PlotLines(usage.c_str(),(const float*)gpuUsage,IM_ARRAYSIZE(gpuUsage),nUsage,text.c_str(),0,total_mem_kb/1024.0);
        }
        std::string fps = std::to_string((int)vizFPS[0]) + " FPS";
        std::string textFps = "FPS";
        ImGui::PlotLines(fps.c_str(),(const float*)vizFPS,IM_ARRAYSIZE(vizFPS),nFPS,textFps.c_str(),0,120.0);

        ImGui::Text("Window size: %d x %d", width, height);

        static bool showProfiler = false;
        static std::string strProfiler = ""; 
        ImGui::Checkbox("Profiler", &showProfiler);
        if(showProfiler) {
            // update profiler string
            if(update % 30 == 0) {
                std::stringstream ss; 
                tk::rt::Profiler::getInstance()->print(ss);
                strProfiler = ss.str();
            }
            ImGui::Text(strProfiler.c_str());
        }

    }else{
        tk::common::Vector3<float> pos = tk::common::tf2pose(drawables[imguiSelected]->tf);
        tk::common::Vector3<float> rot = tk::common::tf2rot(drawables[imguiSelected]->tf);
        std::stringstream ss;
        ss << "tf: \n"<<"\tpos: "<<pos<<"\n";
        ss            <<"\trot: "<<rot<<"\n";

        ImGui::Text("%s", ss.str().c_str());
        ImGui::Separator();
        drawables[imguiSelected]->imGuiInfos();
    }
}

void 
Viewer::drawSettings(){
    if(imguiSelected == -1){
        ImGui::SliderFloat3("Light position",&lightPos[0],0.0f,500.0f);
    }else{
        drawables[imguiSelected]->imGuiSettings();
    }
}

void
Viewer::beforeDraw(){
    for (auto const& drawable : drawables){
        if(drawable.second->enabled)
            drawable.second->beforeDraw(Viewer::instance);
    }
}

void 
Viewer::draw() {
    glfwGetFramebufferSize(window, &width, &height);

    for (auto const& drawable : drawables){
        if(drawable.second->enabled){
            drawable.second->drwModelView = modelview * glm::make_mat4x4(drawable.second->tf.matrix().data());
            drawable.second->draw(Viewer::instance);
        }
        if(drawable.second->enableTf){
            glPushMatrix();
            glMultMatrixf(drawable.second->tf.matrix().data());
            glLineWidth(2.5);
            glBegin(GL_LINES);
            glColor3f (1.0f,0.0f,0.0f);
            glVertex3f(0.0f,0.0f,0.0f);
            glVertex3f(1.0f,0.0f,0.0f);
            glColor3f (0.0f,1.0f,0.0f);
            glVertex3f(0.0f,0.0f,0.0f);
            glVertex3f(0.0f,1.0f,0.0f);
            glColor3f (0.0f,0.0f,1.0f);
            glVertex3f(0.0f,0.0f,0.0f);
            glVertex3f(0.0f,0.0f,1.0f);
            glEnd();
            glPopMatrix();
        }
    }
    drawLogo();
}

void
Viewer::drawLogo(){

    int w=90*3, h=90, padding=20;
    int w2=width/2, h2=height/4;

    verticesCube2D[0] = (float)(w2-padding)/w2;
    verticesCube2D[1] = (float)(-h2+padding)/h2;

    verticesCube2D[5] = (float)(w2-w-padding)/w2;
    verticesCube2D[6] = (float)(-h2+padding)/h2;

    verticesCube2D[10] = (float)(w2-w-padding)/w2;
    verticesCube2D[11] = (float)(-h2+h+padding)/h2;

    verticesCube2D[15] = (float)(w2-padding)/w2;
    verticesCube2D[16] = (float)(-h2+h+padding)/h2;

    pos.setData(verticesCube2D.data(),20);
	pos.setIndexVector(indicesCube2D.data(),6);

    glm::mat4 view2D = glm::mat4(1.0f);
    drwLogo->draw<uint8_t>(view2D,&logo,&pos,6);
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
}

void 
Viewer::follow() {

    tk::common::Vector3<float> center;
    int n = 0;
    for (auto const& drawable : drawables){
        if(drawable.second->follow > 0){
            center += tk::common::tf2pose(drawable.second->tf);
            n++;
        }
    }

    if(n > 0){
        center.x() /= n;
        center.y() /= n;
        center.z() /= n;
        camera.setCenter(center);
    }

    tk::common::Vector3<float> angle;
    n = 0;
    for (auto const& drawable : drawables){
        if(drawable.second->follow > 1){
            angle += tk::common::tf2rot(drawable.second->tf);
            n++;
        }
    }

    if(n > 0){
        angle.x() /= n;
        angle.y() /= n;
        angle.z() /= n;
        camera.setAngle(angle);
    }
}

void 
Viewer::close() { 
    for (auto const& drawable : drawables){
        drawable.second->onClose();
        if(drawable.second->doFree())
            delete drawable.second;
    }   
    drwLogo->close();
    logo.release();
    tkMSG("closed\n");
}

void 
Viewer::runloop() {

    timeStamp_t VIZ_DT_US = dt*1e6;
    tk::rt::Task rate;
    rate.init(VIZ_DT_US);

    while (running) {

        initDrawables();
        beforeDraw();

        // update camera
        follow();
        camera.updateEye();
        camera.updateMatrices();

        glViewport(camera.viewport[0], camera.viewport[1], camera.viewport[2], camera.viewport[3]);

        glClearColor(background.r(), background.g(), background.b(), background.a());
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        ///////Retro compatibility
        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity() ;
        glPushMatrix();
        glMultMatrixf(glm::value_ptr(camera.projection));
        glMultMatrixf(glm::value_ptr(camera.modelView));
        /////////////////////////

        // apply camera matrix
        modelview = camera.projection * camera.modelView;
        
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();
        camera.mouseOnGUI = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);

        //tkRendering
        if(useImGUI == true)
            imguiDraw();
        draw();

        ImGui::Render();
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        ///////Retro compatibility
        glPopMatrix();
        /////////////////////////


        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
        glCheckError();
        running = running && !glfwWindowShouldClose(window);
        
        rate.wait();
    }

    ImPlot::DestroyContext();
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();  
    ImGui::DestroyContext();
    
    glfwDestroyWindow(window);
    glfwTerminate();

    close();
}


void 
Viewer::errorCallback(int error, const char* description) {
    tkERR(std::string{"error: "}+std::to_string(error)+" "+description+"\n");
}

void 
Viewer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
    Viewer::instance->camera.mouseWheel(xoffset, yoffset);
}

void 
Viewer::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    Viewer::instance->camera.mouseMove(xpos, ypos);
}

void 
Viewer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
        Viewer::instance->camera.mouseDown(button, xpos, ypos);
    if (action == GLFW_RELEASE)
        Viewer::instance->camera.mouseUp(button, xpos, ypos);

    for(int i=0; i<Viewer::instance->user_key_callbacks.size(); i++) {
        Viewer::instance->user_key_callbacks[i](button, action, GLFW_SOURCE_MOUSE);
    }
}

void 
Viewer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods){

    if (key == GLFW_KEY_ESCAPE){
        glfwSetWindowShouldClose(window, true); 	
    }

    if(key == GLFW_KEY_R){
        Viewer::instance->camera.init();
    }

    for(int i=0; i<Viewer::instance->user_key_callbacks.size(); i++) {
        Viewer::instance->user_key_callbacks[i](key, action, GLFW_SOURCE_KEYBORAD);
    }
}

void 
Viewer::window_size_callback(GLFWwindow* window, int width, int height){ 
    Viewer::instance->camera.setViewPort(0, 0, width, height);
}


void 
Viewer::addKeyCallback(void (*fun)(int,int,int)) {
    Viewer::instance->user_key_callbacks.push_back(fun);
}

}}

