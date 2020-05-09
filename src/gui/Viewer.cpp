#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/OBJ_Loader.h"

extern bool gRun;
using namespace tk::gui;
bool            Viewer::keys[MAX_KEYS];
Camera3D     Viewer::mouseView;
GLUquadric*     Viewer::quadric;

std::vector<Color_t> tk::gui::Viewer::colors = std::vector<Color_t>{color::RED, color::GREEN, 
                                                                    color::BLUE, color::YELLOW, 
                                                                    color::WHITE, color::ORANGE,
                                                                    color::PINK, color::GREY
                                                                    };
int Viewer::TK_FONT_SIZE = 256;

Viewer::Viewer() {
    // this must stay in the constructor because the file is loaded in the init function
    icon_fname = std::string(std::string(TKPROJ_PATH)+"/data/tkLogo.png");
}


Viewer::~Viewer() {
}

void 
Viewer::init() {
    for(int i=0; i<MAX_KEYS; i++)
        Viewer::keys[i] = false;

    Viewer::quadric = gluNewQuadric();

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

    Viewer::mouseView.window = window;

    glfwSetScrollCallback(window, Viewer::scroll_callback);
    glfwSetCursorPosCallback(window, Viewer::cursor_position_callback);
    glfwSetMouseButtonCallback(window, Viewer::mouse_button_callback);

    glfwSetKeyCallback(window, keyCallback);
    glfwMakeContextCurrent(window);
    //gladLoadGL(glfwGetProcAddress);
    glfwSwapInterval(1);

    //ImGUI
    {
        // Initialize OpenGL loader
        bool err = glewInit() != GLEW_OK;
        if (err) {
            tk::tformat::printErr("Viewer", "Failed to initialize OpenGL loader!\n");
            exit(-1);
        }

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

    // font
    {
        int foo = 1;
        const char* bar[1] = {" "}; 
        glutInit(&foo, (char**)bar);


        /* XXX dtx_open_font opens a font file and returns a pointer to dtx_font */
        if(!(font = dtx_open_font(fontPath.c_str(), TK_FONT_SIZE))) {
            tk::tformat::printErr("Viewer",std::string{"failed to open font: "+fontPath});
            exit(1);
        }
        /* XXX select the font and size to render with by calling dtx_use_font
         * if you want to use a different font size, you must first call:
         * dtx_prepare(font, size) once.
         */
        dtx_use_font(font, TK_FONT_SIZE);

    }

    plotManger = new PlotManager();

    // OPENGL confs
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    //glDepthFunc(GL_GEQUAL);
    //glEnable(GL_LINE_SMOOTH);

    std::string msg =   std::string{"OPENGL running on:"} + 
                        std::string(reinterpret_cast<const char*>(glGetString(GL_VERSION))) + " -- " +
                        std::string(reinterpret_cast<const char*>(glGetString(GL_RENDERER))) + "\n";
    clsMsg(msg);

    tkLoadTexture(std::string(TKPROJ_PATH) + "data/HipertLab.png", hipertTex);
    tkLoadLogo(std::string(TKPROJ_PATH) + "data/tkLogo.pts", logo);

    glfwGetFramebufferSize(window, &width, &height);
    clsSuc("init with resolution " + std::to_string(width) + "x" + std::to_string(height) + "\n");
}


void 
Viewer::draw() {
    tk::common::Vector3<float> s = {1, 1, 1};
    tk::gui::Color_t col = tk::gui::color::LIGHT_BLUE;
    tkSetColor(col);
    tkDrawRectangle(Viewer::mouseView.getPointOnGround(), s, false);
    col.a /= 4;
    tkSetColor(col);
    tkDrawRectangle(Viewer::mouseView.getPointOnGround(), s, true);

    glPushMatrix();
    tk::common::Vector3<float> p = Viewer::mouseView.getWorldPos();
    tkApplyTf(tk::common::odom2tf(p.x, p.y, 0));
    tkDrawAxis();
    glPopMatrix();


}

void
Viewer::drawSplash() {
    // draw 2D HUD
    tkViewport2D(width, height);

    static float x = 0;
    static int j = 0;
    //float size = 0.2f + 0.1*sin(x);

    float y = 0;

    // Fourier series
    for(int i = 1; i <= 3; i+=2){
        y += 1.0/i * sin(2.0 * i *M_PI * x);
    }

    float size;
    // Pulse basing on Fourier series
    if(y>0){
        size= 0.2f + 0.1*y;
        //size = 0.2f + 0.1*(sin(2*x*M_PI));

        // Slower
        x += dt / 4;
    }
    // Collapse in a circle
    else{
        size = 0.2f + 0.1*(sin(2*x*M_PI));

        //Faster
        x += dt / 3;
    }

    //glPushMatrix(); {
    //    tkSetColor(tk::gui::color::WHITE);
    //    tkDrawTexture(hipertTex, size, size);
    //} glPopMatrix();

    glPushMatrix();
    {
        tkSetColor(tk::gui::color::WHITE, size);
        for(int i=0; i<logo.size(); i++)
            tkDrawCircle(tk::common::Vector3<float>(logo[i].x * (4*(size - 0.1)), logo[i].y * (4*(size-0.1)), 0.1), 0.00025 / (size*size*size), 100, true);
            //tkDrawCircle(tk::common::Vector3<float>(logo[i].x * (0.5+size/2), logo[i].y * (0.5+size/2), 0.1), 0.1 * size/2, 100, true);
    }
    glPopMatrix();


}

void Viewer::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) 
{
    Viewer::mouseView.mouseWheel(xoffset, yoffset);
}

void Viewer::cursor_position_callback(GLFWwindow* window, double xpos, double ypos)
{
    Viewer::mouseView.mouseMove(xpos, ypos);
}

void Viewer::mouse_button_callback(GLFWwindow* window, int button, int action, int mods)
{
    double xpos, ypos;
    glfwGetCursorPos(window, &xpos, &ypos);

    if (action == GLFW_PRESS)
        Viewer::mouseView.mouseDown(button, xpos, ypos);
    if (action == GLFW_RELEASE)
        Viewer::mouseView.mouseUp(button, xpos, ypos);
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

        glViewport(0, 0, width, height);

        glMatrixMode( GL_PROJECTION );
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity() ;

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(float(background.r)/255, float(background.g)/255, float(background.b)/255, float(background.a)/255);

        Viewer::mouseView.setWindowAspect(float(width)/height);

        glPushMatrix();

        if(splash) {
            Viewer::mouseView.mouseOnGUI = true;
            drawSplash();
        } else {
            // apply matrix
            glMultMatrixf(Viewer::mouseView.getProjection().data());
            glMultMatrixf(Viewer::mouseView.getModelView().data());

            ImGui_ImplOpenGL3_NewFrame();
            ImGui_ImplGlfw_NewFrame();
            ImGui::NewFrame();

            Viewer::mouseView.mouseOnGUI = ImGui::IsMouseHoveringAnyWindow();

            plotManger->drawPlots();
            draw();

            {
                float plotSize = 0.35;
                int plotW = plotSize*height;
                int plotH = plotSize*height;
                int plotX = width - plotW - 10;
                int plotY = height - plotH - 10;
                tkViewport2D( plotW, plotH, plotX, plotY);
                plotManger->drawLegend();
            }

            // draw tk LOGO
            if(drawLogo) {
                glPushMatrix();
                {
                    tkViewport2D(width, height);
                    float margin = 0.2;
                    tkSetColor(tk::gui::color::WHITE, 0.3);
                    for (int i = 0; i < logo.size(); i++)
                        tkDrawCircle(tk::common::Vector3<float>{
                                logo[i].x * 0.2f + xLim - margin,
                                logo[i].y * 0.2f - yLim + margin,
                                -0.9f}, 0.005, 50, true);

                    tkDrawText("version: " + tk::common::tkVersionGit(),
                            tk::common::Vector3<float>{xLim - margin*1.55f, -yLim + margin/10, -0.9f},
                            tk::common::Vector3<float>{0,0,0},
                            tk::common::Vector3<float>{0.035,0.035,0});
                }
                glPopMatrix();
            }

            ImGui::Render();
            ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());
        }

        glPopMatrix();

        /* Swap front and back buffers */
        glfwSwapBuffers(window);

        /* Poll for and process events */
        glfwPollEvents();
        rate.wait(false);
    }
    gRun = false;
    
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();  
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
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


int 
Viewer::tkLoadTexture(std::string filename, GLuint &tex) {

    unsigned error;
    unsigned char* image;
    unsigned width, height;

    tk::tformat::printMsg("Viewer",std::string{"loading:"+filename}+"\n");

    error = lodepng_decode32_file(&image, &width, &height, filename.c_str());
    if(error == 0) {
        if( (width % 32) != 0 || (height % 32) != 0){
            tk::tformat::printErr("Viewer","please use images size multiple of 32\n");
        }

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
    
    if(error == 0)
        tk::tformat::printSuc("Viewer","file open successfully\n");
    else
        tk::tformat::printErr("Viewer",std::string{"error: "}+std::to_string(error)+std::string{"\n"});

    return error;
}


int 
Viewer::tkLoadOBJ(std::string filename, object3D_t &obj) {

    int error = 0;
    
    tk::tformat::printMsg("Viewer",std::string{"loading:"+filename}+std::string{".obj\n"});
    
    // correct locale dependent stof
    std::setlocale(LC_ALL, "C");

    objl::Loader loader;
    if(loader.LoadFile((filename + ".obj").c_str())) {
   
        tk::tformat::printSuc("Viewer","file open successfully\n");
        obj.triangles.resize(loader.LoadedMeshes.size());
        obj.colors.resize(loader.LoadedMeshes.size());

        for(int o=0; o<loader.LoadedMeshes.size(); o++) {

            //std::string msg = std::string{"name: "}+loader.LoadedMeshes[o].MeshName+"  verts: "+std::to_string(loader.LoadedMeshes[o].Vertices.size())+"\n";
            //tk::tformat::printMsg("Viewer",msg);
            //std::cout<<"name: "<<loader.LoadedMeshes[o].MeshName<<"  verts: "<<loader.LoadedMeshes[o].Vertices.size()<<"\n";
   
            std::vector<unsigned int> indices = loader.LoadedMeshes[o].Indices;
            std::vector<objl::Vertex> verts = loader.LoadedMeshes[o].Vertices;

            obj.colors[o].x = loader.LoadedMeshes[o].MeshMaterial.Kd.X;
            obj.colors[o].y = loader.LoadedMeshes[o].MeshMaterial.Kd.Y;
            obj.colors[o].z = loader.LoadedMeshes[o].MeshMaterial.Kd.Z;

            //msg = std::string{"mat: "}+loader.LoadedMeshes[o].MeshMaterial.name;
            //msg += std::string{" ("}+std::to_string(obj.colors[0].x)+std::to_string(obj.colors[0].y)+std::to_string(obj.colors[0].z)+std::string{")\n"};
            //tk::tformat::printMsg("Viewer",msg);


            //std::cout<<"mat: "<<loader.LoadedMeshes[o].MeshMaterial.name<<" diffuse: "<<obj.colors[o]<<"\n";
            
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


    error = tkLoadTexture((filename + ".png"), obj.tex);
    if(error == 1)
        tk::tformat::printErr("Viewer","Texture not found\n");
    }

    return error;
}

void
Viewer::tkLoadLogo(std::string filename, std::vector<tk::common::Vector3<float>> &logo){
    std::ifstream in;
    in.open(filename);
    float x, y, max_x, max_y;
    in >> x;
    in >> y;
    max_x = x;
    max_y = y;
    while(!in.eof()){
        if(x>max_x)
            max_x = x;
        if(y>max_y)
            max_y = y;
        logo.push_back(tk::common::Vector3<float>(x, y, 0.1));
        in >> x;
        in >> y;
    }

    float max = std::max(max_x, max_y);

    for(int i = 0; i < logo.size(); i++){
        logo[i].x -= max_x/2;
        logo[i].y -= max_y/2;
        logo[i].x /= max;
        logo[i].y /= max;
        logo[i].y *= -1;
    }
    logo.push_back(logo[0]);
    in.close();
}

void 
Viewer::tkSetColor(tk::gui::Color_t c, float alpha) {
    if(alpha >= 0)
        glColor4ub(c.r, c.g, c.b, alpha*255);
    else
        glColor4ub(c.r, c.g, c.b, c.a);
}

void 
Viewer::tkApplyTf(tk::common::Tfpose tf) {
    // apply roto translation
    glMultMatrixf(tf.matrix().data());
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
Viewer::tkDrawTriangle(tk::common::Vector3<float> a, tk::common::Vector3<float> b, tk::common::Vector3<float> c, bool filled) {

    if(filled) {
        glBegin(GL_TRIANGLES);
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(c.x, c.y, c.z);
        glEnd();
    } else {
        glBegin(GL_LINES);
        glVertex3f(a.x, a.y, a.z);
        glVertex3f(b.x, b.y, b.z);
        glVertex3f(c.x, c.y, c.z);
        glVertex3f(a.x, a.y, a.z);
        glEnd();
    }
}


void 
Viewer::tkDrawCircle(tk::common::Vector3<float> pose, float r, int res, bool filled) {

    int res_1 = res;

    if(filled) {
        glBegin(GL_TRIANGLE_FAN);
        glVertex3f(pose.x, pose.y, pose.z);
        res_1++;
    } else {
        glBegin(GL_LINE_LOOP);
    }
    for (int j = 0; j < res_1; j++)   {
        float theta = 2.0f * 3.1415926f * float(j) / float(res);//get the current angle 
        float xr = r * cosf(theta);//calculate the x component 
        float yr = r * sinf(theta);//calculate the y component

        glVertex3f(pose.x + xr, pose.y + yr, pose.z);//output vertex
    }
    glEnd();
}

void 
Viewer::tkDrawSphere(tk::common::Vector3<float> pose, float r, int res, bool filled) {
    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;
    glPushMatrix();
    glTranslatef(pose.x, pose.y, pose.z);
    gluSphere(Viewer::quadric, r, res, res);
    glPopMatrix();

    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL ) ;
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

void Viewer::tkRainbowColor(float hue, uint8_t &r, uint8_t &g, uint8_t &b) {

    hue = fmod(fabs(hue), 1);

    int h = int(hue * 256 * 6);
    int x = h % 0x100;

    switch (h / 256)
    {
    case 0: r = 255; g = x;       break;
    case 1: g = 255; r = 255 - x; break;
    case 2: g = 255; b = x;       break;
    case 3: b = 255; g = 255 - x; break;
    case 4: b = 255; r = x;       break;
    case 5: r = 255; g = x;       break;
    }
}


void Viewer::tkSetRainbowColor(float hue) {
    
    uint8_t r = 0, g = 0, b = 0;
    tkRainbowColor(hue, r, g, b);
    glColor3f(float(r)/255.0, float(g)/255.0, float(b)/255.0);
}


void
Viewer::tkDrawCloudFeatures(Eigen::MatrixXf *points, Eigen::MatrixXf *features, int idx, float maxval) {
    glBegin(GL_POINTS);
    for (int p = 0; p < points->cols(); p++) {
        float i = float(features->coeff(idx, p))/maxval;
        tkSetRainbowColor(i);

        Eigen::Vector4f v = points->col(p);
        glVertex3f(v(0), v(1), v(2));
    }
    glEnd();
}

void
Viewer::tkDrawCloudRGB(Eigen::MatrixXf *points, Eigen::MatrixXf *features, int r, int g, int b) {
    glBegin(GL_POINTS);
    for (int p = 0; p < points->cols(); p++) {
        if(features->coeff(r, p) == 0 && features->coeff(g, p) == 0 && features->coeff(b, p) == 0)
            continue;
        glColor3f(fabs(features->coeff(r, p)), fabs(features->coeff(g, p)), fabs(features->coeff(b, p)));

        Eigen::Vector4f v = points->col(p);
        glVertex3f(v(0), v(1), v(2));
    }
    glEnd();
}

void 
Viewer::tkDrawArrow(float length, float radius, int nbSubdivisions) {
    if (radius < 0.0)
        radius = 0.05 * length;

    const float head = 2.5 * (radius / length) + 0.1;
    const float coneRadiusCoef = 4.0 - 5.0 * head;

    gluCylinder(Viewer::quadric, radius, radius, length * (1.0 - head / coneRadiusCoef), nbSubdivisions, 1);
    glTranslated(0.0, 0.0, length * (1.0 - head));
    gluCylinder(Viewer::quadric, coneRadiusCoef * radius, 0.0, head * length, nbSubdivisions, 1);
    glTranslated(0.0, 0.0, -length * (1.0 - head));
}

void 
Viewer::tkDrawArrow(tk::common::Vector3<float> pose, float yaw, float lenght, float radius, int nbSubdivisions) {
    glPushMatrix();
    glTranslatef(pose.x, pose.y, pose.z);
    glRotatef(90.0, 1.0, 0.0, 0.0);
    glRotatef(90.0 + yaw*180/M_PI, 0.0, 1.0, 0.0);
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
Viewer::tkDrawRectangle(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled) {
    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;

    glPushMatrix();
    glTranslatef(pose.x, pose.y, pose.z);
    glScalef(size.x, size.y, size.z);

    glBegin(GL_POLYGON);
    glVertex3f(  0.5, -0.5, 0 );
    glVertex3f(  0.5,  0.5, 0 );
    glVertex3f( -0.5,  0.5, 0 );
    glVertex3f( -0.5, -0.5, 0 );
    glEnd();

    glPopMatrix();

    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL ) ;
}

void Viewer::tkDrawLine(tk::common::Vector3<float> p0, tk::common::Vector3<float> p1) {
    glBegin(GL_LINES);
    glVertex3f(p0.x, p0.y, p0.z);
    glVertex3f(p1.x, p1.y, p1.z);
    glEnd();
}
void Viewer::tkDrawLine(std::vector<tk::common::Vector3<float>> poses) {
    glBegin(GL_LINE_STRIP);
    for(unsigned int i=0; i<poses.size(); i++) {
        glVertex3f(poses[i].x, poses[i].y, poses[i].z);
    }
    glEnd();
}

void Viewer::tkDrawPoses(std::vector<tk::common::Vector3<float>> poses, tk::common::Vector3<float> size) {
    for(unsigned int i=0; i<poses.size(); i++) {
        tkDrawCube(poses[i], size);
    }
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

void Viewer::tkDrawTexture(GLuint tex, float sx, float sy) {

    float i = -1;
    float j = +1;

    glBindTexture(GL_TEXTURE_2D, tex);
    glEnable(GL_TEXTURE_2D);
    glBegin(GL_QUADS);

    // 2d draw
    glTexCoord2f(0, 1); glVertex3f(i*(sy/2), i*(sx/2), 0);
    glTexCoord2f(0, 0); glVertex3f(i*(sy/2), j*(sx/2), 0);
    glTexCoord2f(1, 0); glVertex3f(j*(sy/2), j*(sx/2), 0);
    glTexCoord2f(1, 1); glVertex3f(j*(sy/2), i*(sx/2), 0);


    glEnd();
    glDisable(GL_TEXTURE_2D);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void
Viewer::tkDrawText(std::string text, tk::common::Vector3<float> pose, tk::common::Vector3<float> rot, tk::common::Vector3<float> size) {

    float corectRatio = 1.0/TK_FONT_SIZE;
    
    glPushMatrix();
    glTranslatef(pose.x, pose.y, pose.z);
    glRotatef(rot.x*180.0/M_PI, 1, 0, 0);
    glRotatef(rot.y*180.0/M_PI, 0, 1, 0);
    glRotatef(rot.z*180.0/M_PI, 0, 0, 1);    
    glScalef(size.x*corectRatio, size.y*corectRatio, size.z*corectRatio);

    dtx_string(text.c_str());
    //glutStrokeString(GLUT_STROKE_ROMAN, (unsigned char*) text.c_str());
    
    glPopMatrix();
}

void 
Viewer::tkDrawRadarData(tk::data::RadarData *data) {
    tk::common::Vector3<float> pose;
    tk::common::Tfpose  correction = tk::common::odom2tf(0, 0, 0, +M_PI/2);
    for(int i = 0; i < data->nRadar; i++) {
        glPushMatrix();
        tkDrawTf(data->near_data[i].header.name, (data->near_data[i].header.tf * correction));
        tkApplyTf(data->near_data[i].header.tf);
        // draw near
        for (int j = 0; j < data->near_data[i].nPoints; j++) {
            float rcs = data->near_features[i](tk::data::RadarFeatureType::RCS, j);

            //NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
            float hue = (((rcs + 40) * (1 - 0)) / (20 + 40)) + 0;
            tkSetRainbowColor(hue);

            pose.x = data->near_data[i].points(0, j);
            pose.y = data->near_data[i].points(1, j);
            pose.z = data->near_data[i].points(2, j);

            tkDrawCircle(pose, 0.05);
        }
        //// draw far
        //for (int j = 0; j < data->far_data[i].nPoints; j++) {
        //    float rcs = data->far_data[i].features(tk::data::CloudFeatureType::RCS, j);
//
        //    //NewValue = (((OldValue - OldMin) * (NewMax - NewMin)) / (OldMax - OldMin)) + NewMin
        //    float hue = (((rcs + 40) * (1 - 0)) / (20 + 40)) + 0;
        //    tkSetRainbowColor(hue);
//
        //    pose.x = data->far_data[i].points(0, j);
        //    pose.y = data->far_data[i].points(1, j);
        //    pose.z = data->far_data[i].points(2, j);
//
        //    tkDrawCircle(pose, 0.05);
        //}
        glPopMatrix();
    }
}

void
Viewer::tkDrawLiDARData(tk::data::LidarData *data){

    glPointSize(1.0);
    glBegin(GL_POINTS);
    //white
    tkSetColor(tk::gui::color::WHITE);

    for (int p = 0; p < data->nPoints; p++) {
        float i = float(data->intensity(p));
        tkSetRainbowColor(i);
        glVertex3f(data->points.coeff(0,p),data->points.coeff(1,p),data->points.coeff(2,p));
    }
    glEnd();
}

void
Viewer::tkDrawImage(tk::data::ImageData<uint8_t>& image, GLuint texture)
{

    if(image.empty()){
        tk::tformat::printMsg("Viewer","Image empty\n");
    }else{

        //glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
        glBindTexture(GL_TEXTURE_2D, texture);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

        // Set texture clamping method
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);

        if(image.channels == 4) {
            glTexImage2D(GL_TEXTURE_2D,         // Type of texture
                         0,                   // Pyramid level (for mip-mapping) - 0 is the top level
                         GL_RGB,              // Internal colour format to convert to
                         image.width,          // Image width  i.e. 640 for Kinect in standard mode
                         image.height,          // Image height i.e. 480 for Kinect in standard mode
                         0,                   // Border width in pixels (can either be 1 or 0)
                         GL_RGBA,              // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                         GL_UNSIGNED_BYTE,    // Image data type
                         image.data);        // The actual image data itself
        }else if(image.channels == 3){
            glTexImage2D(GL_TEXTURE_2D,         // Type of texture
                         0,                   // Pyramid level (for mip-mapping) - 0 is the top level
                         GL_RGB,              // Internal colour format to convert to
                         image.width,          // Image width  i.e. 640 for Kinect in standard mode
                         image.height,          // Image height i.e. 480 for Kinect in standard mode
                         0,                   // Border width in pixels (can either be 1 or 0)
                         GL_RGB,              // Input image format (i.e. GL_RGB, GL_RGBA, GL_BGR etc.)
                         GL_UNSIGNED_BYTE,    // Image data type
                         image.data);        // The actual image data itself
        }
    }
}


void
Viewer::tkSplitPanel(int count, float ratio, float xLim, int &num_cols, int &num_rows, float &w, float &h, float &x, float &y){
    num_rows = 4;
    if(ratio <= 0) {
        num_rows = ceil(sqrt(count));
    }else{
        num_rows = count > 4 ? num_rows : 4;
        num_rows = count > 8 ? 8 : num_rows;
    }
    num_cols = ceil((float)count/num_rows);

    h = 1.0f/((float)num_rows/2);
    if(ratio > 0){
        w = h * ratio;
    }
    else{
        w = 2.0f / num_cols;
    }

    if(ratio > 0){
        x = -xLim + w/2 + w * (num_cols-1);
        y = -1.0f + h/2;
    }
    else {
        x = -w * ((float) num_cols / 2) + w / 2;
        y = -1.0f + h / 2;
    }
}

void
Viewer::tkDrawSpeedometer(tk::common::Vector2<float> pose, float speed, float radius) {
    glPushMatrix(); 

    float diff_radius = 0.03;
    float outer_radius = radius;
    float inner_radius = radius - diff_radius;
    
    tk::common::Vector2<float> indicator_a, indicator_b, indicator_c, indicator_d;
    float indicator_angle = float(speed * 3.6); 

    

    float D2R = 0.0174532;

    Color_t col = tk::gui::color::DARK_GRAY; col.a = 185;
    tkSetColor(col);
    tkDrawCircle(tk::common::Vector3<float>{pose.x, pose.y, 0.2}, radius, 72, true);

    // outer circle
    tkSetColor(tk::gui::color::WHITE);
    glLineWidth(1);
	glBegin(GL_LINE_LOOP);
	for (float angle = 0; angle < 360; angle += 5) {
		glVertex2f(outer_radius * cos(angle*D2R) + pose.x, outer_radius * sin(angle*D2R) + pose.y);
	}
	glEnd();

    // inner circle
    tkSetColor(tk::gui::color::WHITE);
	glBegin(GL_LINE_STRIP);
	glVertex2f((inner_radius + 0.005) * cos(-70 * D2R) + pose.x, (inner_radius + 0.005) * sin(-70 * D2R) + pose.y);
	glVertex2f((outer_radius) * cos(-70 * D2R) + pose.x, (outer_radius) * sin(-70 * D2R) + pose.y);
	for (float angle = 0 - 70; angle <= 320 - 70; angle += 5) {
		glVertex2f(inner_radius * cos(angle*D2R) + pose.x, inner_radius * sin(angle*D2R) + pose.y);
	}
	glVertex2f((inner_radius + 0.005) * cos(250 * D2R) + pose.x, (inner_radius + 0.005) * sin(250 * D2R) + pose.y);
	glVertex2f((outer_radius) * cos(250 * D2R) + pose.x, (outer_radius) * sin(250 * D2R) + pose.y);
	glEnd();

    // speed text lines
    tkSetColor(tk::gui::color::WHITE);
	glBegin(GL_LINES);
	for (float angle = 0 - 70; angle <= 320 - 70; angle += 20) {
		glVertex2f(inner_radius * cos(angle*D2R) + pose.x, inner_radius * sin(angle*D2R) + pose.y);
		glVertex2f((inner_radius - 0.02) * cos(angle*D2R) + pose.x, (inner_radius - 0.02) * sin(angle*D2R) + pose.y);
	}
	glEnd();

    // speed text
    /*
    std::string speedText;
    for (float angle = 0 - 70; angle <= 320 - 70; angle += 20) {
        float x, y, z;

        x = (radius - 0.36) * cos(angle*D2R) + pose.x;
        y = (radius - 0.36) * sin(angle*D2R) + pose.y;
        z = 0;
        speedText = std::to_string(int(-1 * (angle - 250)));
        tkDrawText(speedText, tk::common::Vector3<float>{x, y, z},
                              tk::common::Vector3<float>{0, 0, 0},
                              tk::common::Vector3<float>{0.02, 0.02, 0.0});
	}
    */
    tkSetColor(tk::gui::color::YELLOW);
    char speedStr[256];
    sprintf(speedStr, "%.2f", speed/3.6);
    tkDrawText(speedStr, tk::common::Vector3<float>{pose.x, pose.y - 0.08f, 0},
                         tk::common::Vector3<float>{0, 0, 0},
                         tk::common::Vector3<float>{0.05f, 0.05f, 0.0f});
    tkDrawText("km/h", tk::common::Vector3<float>{pose.x, pose.y - 0.12f, 0},
                       tk::common::Vector3<float>{0, 0, 0},
                       tk::common::Vector3<float>{0.03f, 0.03f, 0.0f});

    // danger zone
    tkSetColor(tk::gui::color::RED);
    glLineWidth(4);
	glBegin(GL_LINE_STRIP);
	for (float angle = 320 - 30; angle <= 320 + 30; angle += 5) {
		glVertex2f((inner_radius - 0.01) * cos(angle*D2R) + pose.x, (inner_radius - 0.01) * sin(angle*D2R) + pose.y);
	}
	glEnd();

    // indicator
    indicator_a.x = inner_radius * cos((250 + -indicator_angle) * D2R) + pose.x;
	indicator_a.y = inner_radius * sin((250 + -indicator_angle) * D2R) + pose.y;
	indicator_b.x = 0.01875 * cos((150 + -indicator_angle) * D2R) + pose.x;
	indicator_b.y = 0.01875 * sin((150 + -indicator_angle) * D2R) + pose.y;
	indicator_c.x = 0.025 * cos((70 + -indicator_angle) * D2R) + pose.x;
	indicator_c.y = 0.025 * sin((70 + -indicator_angle) * D2R) + pose.y;
	indicator_d.x = 0.01875 * cos((-10 + -indicator_angle) * D2R) + pose.x;
	indicator_d.y = 0.01875 * sin((-10 + -indicator_angle) * D2R) + pose.y;

    glBegin(GL_QUADS);
	glVertex2f(indicator_a.x, indicator_a.y);
	glVertex2f(indicator_b.x, indicator_b.y);
	glVertex2f(indicator_c.x, indicator_c.y);
	glVertex2f(indicator_d.x, indicator_d.y);
	glEnd();

	glLineWidth(3);
	glColor3f(0.5, 0, 0);
	glBegin(GL_LINE_LOOP);
	glVertex2f(indicator_a.x, indicator_a.y);
	glVertex2f(indicator_b.x, indicator_b.y);
	glVertex2f(indicator_c.x, indicator_c.y);
	glVertex2f(indicator_d.x, indicator_d.y);
	glEnd();

    tkSetColor(tk::gui::color::BLACK);
    tkDrawSphere(tk::common::Vector3<float>{pose.x, pose.y, 0}, 0.01);

    // speed arc
    tkSetColor(tk::gui::color::GREEN);
	if (indicator_angle > 260)
		tkSetColor(tk::gui::color::RED);
	glLineWidth(4);
	glBegin(GL_LINE_STRIP);
	glVertex2f((radius - 0.0075) * cos(250 * D2R) + pose.x, (radius - 0.0075) * sin(250 * D2R) + pose.y);
	glVertex2f((radius - 0.0225) * cos(250 * D2R) + pose.x, (radius - 0.0225) * sin(250 * D2R) + pose.y);
	for (float a = 0; a <= indicator_angle; a += 5) {
		glVertex2f((radius - 0.0225) * cos((250 + -a)*D2R) + pose.x, (radius - 0.0225) * sin((250 + -a)*D2R) + pose.y);
	}
	glVertex2f((radius - 0.0225) * cos((250 + -indicator_angle) * D2R) + pose.x, (radius - 0.0225) * sin((250 + -indicator_angle) * D2R) + pose.y);
	glVertex2f((radius - 0.0075) * cos((250 + -indicator_angle) * D2R) + pose.x, (radius - 0.0075) * sin((250 + -indicator_angle) * D2R) + pose.y);
	glEnd();
	glBegin(GL_LINE_STRIP);
	for (float a = 0; a <= indicator_angle; a++) {
		glVertex2f((radius - 0.0075) * cos((250 + -a)*D2R) + pose.x, (radius - 0.0075) * sin((250 + -a)*D2R) + pose.y);
	}
	glEnd();


    glPopMatrix();
    glLineWidth(1);
}

void 
Viewer::tkViewport2D(int width, int height, int x, int y) {
    float ar = (float)width / (float)height;

    glViewport(x, y, width, height);
    glOrtho(0, width, 0, height, -1, 1);
    glLoadIdentity();
    
    glMatrixMode( GL_PROJECTION );
    glLoadIdentity();

    glOrtho(-ar, ar, -1, 1, 1, -1);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void
Viewer::tkDrawTf(std::string name, tk::common::Tfpose tf) {
    // draw tfs
    tk::gui::Viewer::tkSetColor(tk::gui::color::WHITE);
    glLineWidth(1);
    tk::gui::Viewer::tkDrawLine(tk::common::Vector3<float>{0,0,0}, tk::common::tf2pose(tf));
    glPushMatrix();
    tk::gui::Viewer::tkApplyTf(tf);
    tk::gui::Viewer::tkDrawText(name,
                                tk::common::Vector3<float>{0.01,0.01,0},
                                tk::common::Vector3<float>{M_PI/2,0,0},
                                tk::common::Vector3<float>{0.08,0.08,0});
    tk::gui::Viewer::tkDrawAxis(0.1);
    glPopMatrix();
}

void 
Viewer::errorCallback(int error, const char* description) {
    tk::tformat::printErr("Viewer", std::string{"error: "}+std::to_string(error)+" "+description+"\n");
}

void
Viewer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods) {
    if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);

    if (key < MAX_KEYS) {
        if(action == GLFW_PRESS)
            keys[key] = true;
        if(action == GLFW_RELEASE)
            keys[key] = false;
    }
}









