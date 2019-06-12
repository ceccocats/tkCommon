#include "tkCommon/gui/Viewer.h"

using namespace tk::gui;
MouseView3D     mouseView;

Viewer::Viewer() {
}


Viewer::~Viewer() {
}

void 
Viewer::init() {
    glfwSetErrorCallback(errorCallback);
    glfwInit();

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(640, 480, windowName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
    }
    mouseView.window = window;

    glfwSetScrollCallback(window, Viewer::scroll_callback);
    glfwSetCursorPosCallback(window, Viewer::cursor_position_callback);
    glfwSetMouseButtonCallback(window, Viewer::mouse_button_callback);

    glfwSetKeyCallback(window, keyCallback);
    glfwMakeContextCurrent(window);
    //gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);



    // Initialize OpenGL loader
#if defined(IMGUI_IMPL_OPENGL_LOADER_GL3W)
    bool err = gl3wInit() != 0;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLEW)
    bool err = glewInit() != GLEW_OK;
#elif defined(IMGUI_IMPL_OPENGL_LOADER_GLAD)
    bool err = gladLoadGL() == 0;
#else
    bool err = false; // If you use IMGUI_IMPL_OPENGL_LOADER_CUSTOM, your loader is likely to requires some form of initialization.
#endif
    if (err) {
        fprintf(stderr, "Failed to initialize OpenGL loader!\n");
        exit(-1);
    }

    // Setup Dear ImGui context
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;     // Enable Keyboard Controls
    //io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls

    // Setup Dear ImGui style
    ImGui::StyleColorsDark();
    //ImGui::StyleColorsClassic();

    // Setup Platform/Renderer bindings
    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);




    // OPENGL confs
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    //glDepthFunc(GL_GEQUAL);
}


void 
Viewer::draw() {

    tk::common::Vector3<float> s = { 1, 1, 0 };
    tk::gui::Color_t col = tk::gui::color::LIGHT_BLUE;
    tkSetColor(col);
    tkDrawCube(mouseView.getPointOnGround(), s, false);
    col.a /= 4;
    tkSetColor(col);
    tkDrawCube(mouseView.getPointOnGround(), s, true);

    glPushMatrix();
    tk::common::Vector3<float> p = mouseView.getWorldPos();
    tkApplyTf(tk::common::odom2tf(p.x, p.y, 0));
    tkDrawAxis();
    glPopMatrix();
}

void Viewer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) 
{
    mouseView.mouseWheel(xoffset, yoffset);
}

void Viewer::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    mouseView.mouseMove(xpos, ypos);
}

void Viewer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
        mouseView.mouseDown(button, xpos, ypos);
    if (action == GLFW_RELEASE)
        mouseView.mouseUp(button, xpos, ypos);
}

void 
Viewer::run() {
    bool show_demo_window = true;
    LoopRate rate((1e6/60), "VIZ_UPDATE");
    while (!glfwWindowShouldClose(window)) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        // 1. Show the big demo window (Most of the sample code is in ImGui::ShowDemoWindow()! You can browse its code to learn more about Dear ImGui!).
        if (show_demo_window)
            ImGui::ShowDemoWindow(&show_demo_window);

        ImGui::Render();
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(float(background.r)/255, float(background.g)/255, float(background.b)/255, float(background.a)/255);

        mouseView.setWindowAspect(float(width)/height);


        glPushMatrix();

        // apply matrix
        glMultMatrixf(mouseView.getProjection()->data());
        glMultMatrixf(mouseView.getModelView()->data());

        draw();

        glPopMatrix();


        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
        rate.wait();
    }
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();  
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
}


void
Viewer::setWindowName(std::string name) {
    windowName = name;
}


void 
Viewer::setBackground(tk::gui::Color_t c) {
    background = c;
}


int 
Viewer::tkLoadTexture(std::string filename, GLuint &tex) {

    unsigned error;
    unsigned char* image;
    unsigned width, height;

    error = lodepng_decode32_file(&image, &width, &height, filename.c_str());
    if(error == 0) {
        if( (width % 32) != 0 || (height % 32) != 0)
            std::cout<<"please use images size multiple of 32\n";

        //upload to GPU texture
        glGenTextures(1, &tex);
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, image);
        glBindTexture(GL_TEXTURE_2D, 0);

        delete [] image;

        if(tex == 0)
            error = 1;
    } else {
        tex = 0;
    }

    std::cout<<"loading "<<filename<<"    ";
    if(error == 0)
        std::cout<<" OK!!\n";
    else
        std::cout<<" ERROR: "<<error<<"\n";

    return error;
}


int 
Viewer::tkLoadOBJ(std::string filename, object3D_t &obj) {

    int error = 0;
    std::cout<<"loading "<<filename<<".obj    ";
    
    // correct locale dependent stof
    std::setlocale(LC_ALL, "C");

    objl::Loader loader;
    if(loader.LoadFile((filename + ".obj").c_str())) {
   
        std::cout<<" OK!!\n";
        obj.triangles.resize(loader.LoadedMeshes.size());
        obj.colors.resize(loader.LoadedMeshes.size());

        for(int o=0; o<loader.LoadedMeshes.size(); o++) {
            std::cout<<"name: "<<loader.LoadedMeshes[o].MeshName<<"  verts: "<<loader.LoadedMeshes[o].Vertices.size()<<"\n";
   
            std::vector<unsigned int> indices = loader.LoadedMeshes[o].Indices;
            std::vector<objl::Vertex> verts = loader.LoadedMeshes[o].Vertices;

            obj.colors[o].x = loader.LoadedMeshes[o].MeshMaterial.Kd.X;
            obj.colors[o].y = loader.LoadedMeshes[o].MeshMaterial.Kd.Y;
            obj.colors[o].z = loader.LoadedMeshes[o].MeshMaterial.Kd.Z;
            std::cout<<"mat: "<<loader.LoadedMeshes[o].MeshMaterial.name<<" diffuse: "<<obj.colors[o]<<"\n";
            
            obj.triangles[o] = Eigen::MatrixXf(5, indices.size());
            for(int i=0; i<indices.size(); i++) {
                int idx = indices[i];
                obj.triangles[o](0,i) = verts[idx].Position.X;
                obj.triangles[o](1,i) = verts[idx].Position.Y;
                obj.triangles[o](2,i) = verts[idx].Position.Z;
                obj.triangles[o](3,i) = verts[idx].TextureCoordinate.X;
                obj.triangles[o](4,i) = 1 - verts[idx].TextureCoordinate.Y;
            }
        }


    } else {
        error = 1;
        std::cout<<" ERROR\n";
    }
    
    error = error || tkLoadTexture((filename + ".png"), obj.tex);
    return error;
}

void 
Viewer::tkSetColor(tk::gui::Color_t c) {
    glColor4ub(c.r, c.g, c.b, c.a);
}

void 
Viewer::tkApplyTf(tk::common::Tfpose tf) {
    // apply roto translation
    tk::common::Vector3<float> p = tk::common::tf2pose(tf);
    tk::common::Vector3<float> r = tk::common::tf2rot (tf);
    glTranslatef(p.x, p.y, p.z);
    glRotatef(r.x*180.0/M_PI, 1, 0, 0);
    glRotatef(r.y*180.0/M_PI, 0, 1, 0);
    glRotatef(r.z*180.0/M_PI, 0, 0, 1);
}  

void 
Viewer::tkDrawAxis(float s) {
    glLineWidth(2.5); 
    glBegin(GL_LINES);
    // x
    glColor3f(1.0, 0.0, 0.0);
    glVertex3f(0, 0, 0);
    glVertex3f(s, 0, 0);
    // y
    glColor3f(0.0, 1.0, 0.0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, s, 0);
    // z
    glColor3f(0.0, 0.0, 1.0);
    glVertex3f(0, 0, 0);
    glVertex3f(0, 0, s);
    glEnd();
}

void 
Viewer::tkDrawCircle(float x, float y, float z, float r, int res) {
    glBegin(GL_LINE_LOOP);
    for (int j = 0; j < res; j++)   {
        float theta = 2.0f * 3.1415926f * float(j) / float(res);//get the current angle 
        float xr = r * cosf(theta);//calculate the x component 
        float yr = r * sinf(theta);//calculate the y component 
        glVertex3f(x + xr, y + yr, z);//output vertex
    }
    glEnd(); 
}

void 
Viewer::tkDrawCloud(Eigen::MatrixXf *points) {
    glBegin(GL_POINTS);
    for (int p = 0; p < points->cols(); p++) {
        Eigen::Vector4f v = points->col(p);
        glVertex3f(v(0), v(1), v(2));
    }
    glEnd();
}

void 
Viewer::tkDrawArrow(float length, float radius, int nbSubdivisions) {
    
    /*static GLUquadric *quadric = gluNewQuadric();

    if (radius < 0.0)
        radius = 0.05 * length;

    const float head = 2.5 * (radius / length) + 0.1;
    const float coneRadiusCoef = 4.0 - 5.0 * head;

    gluCylinder(quadric, radius, radius, length * (1.0 - head / coneRadiusCoef),
                nbSubdivisions, 1);
    glTranslated(0.0, 0.0, length * (1.0 - head));
    gluCylinder(quadric, coneRadiusCoef * radius, 0.0, head * length,
                nbSubdivisions, 1);
    glTranslated(0.0, 0.0, -length * (1.0 - head));*/
}

void 
Viewer::tkDrawArrow(tk::common::Vector3<float> pose, float yaw, float lenght, float radius, int nbSubdivisions) {
    glPushMatrix();
    glRotatef(90.0, 1.0, 0.0, 0.0);
    glRotatef(90.0 + yaw*180/M_PI, 0.0, 1.0, 0.0);
    glTranslatef(pose.x, pose.y, pose.z);
    tkDrawArrow(lenght, radius, nbSubdivisions);
    glPopMatrix();
}

void 
Viewer::tkDrawCube(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled) {
    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;

    glPushMatrix();
    glTranslatef(pose.x, pose.y, pose.z);
    glScalef(size.x, size.y, size.z);

    // BACK
    glBegin(GL_POLYGON);
    glVertex3f(  0.5, -0.5, 0.5 );
    glVertex3f(  0.5,  0.5, 0.5 );
    glVertex3f( -0.5,  0.5, 0.5 );
    glVertex3f( -0.5, -0.5, 0.5 );
    glEnd();
    
    // RIGHT
    glBegin(GL_POLYGON);
    glVertex3f( 0.5, -0.5, -0.5 );
    glVertex3f( 0.5,  0.5, -0.5 );
    glVertex3f( 0.5,  0.5,  0.5 );
    glVertex3f( 0.5, -0.5,  0.5 );
    glEnd();
    
    // LEFT
    glBegin(GL_POLYGON);
    glVertex3f( -0.5, -0.5,  0.5 );
    glVertex3f( -0.5,  0.5,  0.5 );
    glVertex3f( -0.5,  0.5, -0.5 );
    glVertex3f( -0.5, -0.5, -0.5 );
    glEnd();
    
    // TOP
    glBegin(GL_POLYGON);
    glVertex3f(  0.5,  0.5,  0.5 );
    glVertex3f(  0.5,  0.5, -0.5 );
    glVertex3f( -0.5,  0.5, -0.5 );
    glVertex3f( -0.5,  0.5,  0.5 );
    glEnd();
    
    // BOTTOM
    glBegin(GL_POLYGON);
    glVertex3f(  0.5, -0.5, -0.5 );
    glVertex3f(  0.5, -0.5,  0.5 );
    glVertex3f( -0.5, -0.5,  0.5 );
    glVertex3f( -0.5, -0.5, -0.5 );
    glEnd();

    glPopMatrix();

    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL ) ;
}

void 
Viewer::tkDrawObject3D(object3D_t *obj, float size, bool textured) {

    glPushMatrix();
    glScalef(size, size, size);

    if(textured) {
        glBindTexture(GL_TEXTURE_2D, obj->tex);
        glEnable(GL_TEXTURE_2D);
    }

    glBegin(GL_TRIANGLES);
    for(int o=0; o<obj->triangles.size(); o++) {
        if(!textured)
            glColor3f(obj->colors[o].x, obj->colors[o].y, obj->colors[o].z);

        for(int i=0; i<obj->triangles[o].cols(); i++) {
            glTexCoord2f(obj->triangles[o](3,i), obj->triangles[o](4,i)); 
            glVertex3f(obj->triangles[o](0,i),obj->triangles[o](1,i),obj->triangles[o](2,i));
        }
    }
    glEnd();

    if(textured) {
        glDisable(GL_TEXTURE_2D);
        glBindTexture(GL_TEXTURE_2D, 0);
    }
    glPopMatrix();
}

void Viewer::tkDrawTexture(GLuint tex, float s) {

    float i = -s/2;
    float j = +s/2;

    glBindTexture(GL_TEXTURE_2D, tex);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);

    // 2d draw
    glTexCoord2f(0, 1); glVertex3f(i, i, 0);
    glTexCoord2f(0, 0); glVertex3f(i, j, 0);
    glTexCoord2f(1, 0); glVertex3f(j, j, 0);
    glTexCoord2f(1, 1); glVertex3f(j, i, 0);


    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}


void Viewer::tkViewport2D(int width, int height) {
    glViewport(0, 0, width, height);
    glOrtho(0, width, 0, height, -1, 1);
    glLoadIdentity();
 }

void 
Viewer::errorCallback(int error, const char* description) {
    fprintf(stderr, "Error: %s\n", description);
}

void
Viewer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
}









