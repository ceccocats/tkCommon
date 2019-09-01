#include "tkCommon/gui/Viewer.h"
#include "tkCommon/gui/OBJ_Loader.h"

using namespace tk::gui;
bool            Viewer::keys[MAX_KEYS];
MouseView3D     Viewer::mouseView;
GLUquadric*     Viewer::quadric;

std::vector<Color_t> tk::gui::Viewer::colors = std::vector<Color_t>{color::RED, color::PINK, 
                                                                    color::BLUE, color::YELLOW, 
                                                                    color::WHITE, color::ORANGE
                                                                    };
int Viewer::TK_FONT_SIZE = 256;

Viewer::Viewer() {
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

    // Create a windowed mode window and its OpenGL context
    window = glfwCreateWindow(width, height, windowName.c_str(), NULL, NULL);
    if (!window) {
        glfwTerminate();
    }
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
            fprintf(stderr, "Failed to initialize OpenGL loader!\n");
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
        char* bar[1] = {" "}; 
        glutInit(&foo, bar);


        /* XXX dtx_open_font opens a font file and returns a pointer to dtx_font */
        if(!(font = dtx_open_font(fontPath.c_str(), TK_FONT_SIZE))) {
            std::cout<<"failed to open font: "<<fontPath<<"\n";
            exit(1);
        }
        /* XXX select the font and size to render with by calling dtx_use_font
         * if you want to use a different font size, you must first call:
         * dtx_prepare(font, size) once.
         */
        dtx_use_font(font, TK_FONT_SIZE);

    }
    
    // OPENGL confs
    glDisable(GL_LIGHTING);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_DEPTH_TEST);
    //glDepthFunc(GL_GEQUAL);
    //glEnable(GL_LINE_SMOOTH);
}


void 
Viewer::draw() {

    tk::common::Vector3<float> s = { 1, 1, 1 };
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
    LoopRate rate((1e6/30), "VIZ_UPDATE");
    while (!glfwWindowShouldClose(window)) {
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        Viewer::mouseView.mouseOnGUI = ImGui::IsMouseHoveringAnyWindow(); 
            
        glfwGetFramebufferSize(window, &width, &height);
        glViewport(0, 0, width, height);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        glClearColor(float(background.r)/255, float(background.g)/255, float(background.b)/255, float(background.a)/255);

        Viewer::mouseView.setWindowAspect(float(width)/height);


        glPushMatrix();

        // apply matrix
        glMultMatrixf(Viewer::mouseView.getProjection()->data());
        glMultMatrixf(Viewer::mouseView.getModelView()->data());

        draw();

        glPopMatrix();

        ImGui::Render();
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
    if(hue <= 0.0)
        hue = 0.000001;
    if(hue >= 1.0)
        hue = 0.999999;

    int h = int(hue * 256 * 6);
    int x = h % 0x100;

    switch (h / 256)
    {
    case 0: r = 255; g = x;       break;
    case 1: g = 255; r = 255 - x; break;
    case 2: g = 255; b = x;       break;
    case 3: b = 255; g = 255 - x; break;
    case 4: b = 255; r = x;       break;
    case 5: r = 255; b = 255 - x; break;
    }
}


void Viewer::tkSetRainbowColor(float hue) {
    
    uint8_t r = 0, g = 0, b = 0;
    tkRainbowColor(hue, r, g, b);
    glColor3f(float(r)/255.0, float(g)/255.0, float(b)/255.0);
}


void
Viewer::tkDrawCloudFeatures(Eigen::MatrixXf *points, tk::common::MatrixXu8 *features, int idx) {
    glBegin(GL_POINTS);
    for (int p = 0; p < points->cols(); p++) {
        float i = float(features->coeff(idx, p))/255.0;
        tkSetRainbowColor(i);

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
Viewer::tkDrawRadarData(tk::data::RadarData_t *data, bool enable_near, bool enable_far) {
    glPushMatrix(); 
    if (enable_near) {
        tk::common::Vector3<float> pose;
        for (int i = 0; i < N_RADAR; i++) {
            tkSetColor(colors[i]);
            for (int j = 0; j < data->near_n_points[i]; j++) {
                pose.x = data->near_points[i](0, j);
                pose.y = data->near_points[i](1, j);
                pose.z = data->near_points[i](2, j);

                tkDrawCircle(pose, 0.05);     
            }       
        }
    }
    if (enable_far) {
        tk::common::Vector3<float> pose;
        for (int i = 0; i < N_RADAR; i++) {
            tkSetColor(colors[i]);
            for (int j = 0; j < data->far_n_points[i]; j++) {
                pose.x = data->far_points[i](0, j);
                pose.y = data->far_points[i](1, j);
                pose.z = data->far_points[i](2, j);

                tkDrawCircle(pose, 0.05);     
            }       
        }    
    }
    glPopMatrix();
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
    tkDrawCircle(tk::common::Vector3<float>{pose.x, pose.y, +0.2}, radius, 72, true);

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
    tkDrawText(speedStr, tk::common::Vector3<float>{pose.x, pose.y - 0.08, 0},
                         tk::common::Vector3<float>{0, 0, 0},
                         tk::common::Vector3<float>{0.05, 0.05, 0.0});
    tkDrawText("km/h", tk::common::Vector3<float>{pose.x, pose.y - 0.12, 0},
                       tk::common::Vector3<float>{0, 0, 0},
                       tk::common::Vector3<float>{0.03, 0.03, 0.0});

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
    glViewport(x, y, width, height);
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

    if (key < MAX_KEYS) {
        if(action == GLFW_PRESS)
            keys[key] = true;
        if(action == GLFW_RELEASE)
            keys[key] = false;
    }
}









