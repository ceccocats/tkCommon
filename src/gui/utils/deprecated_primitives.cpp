#include <GL/glew.h> 
#include <GL/freeglut.h>
#include <GLFW/glfw3.h>
#include "tkCommon/gui/utils/deprecated_primitives.h"

namespace tk { namespace gui { namespace prim {

void drawAxis(float s, float lw) {
    glLineWidth(lw); 
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
    glLineWidth(1.0); 
}


void drawTexture(GLuint tex, float sx, float sy) {

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

void genTexture(uint8_t *data, int width, int height, int channels, GLuint &tex) {
    int step = width*channels*sizeof(uint8_t);
    //upload to GPU texture
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    //use fast 4-byte alignment (default anyway) if possible
    glPixelStorei(GL_UNPACK_ALIGNMENT, (step & 3) ? 1 : 4);
    //set length of one complete row in data (doesn't need to equal image.cols)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, step/channels);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void drawCircle(tk::common::Vector3<float> pose, float r, int res, bool filled) {
    drawEllipse(pose, r, r, res, filled);
}

void drawEllipse(tk::common::Vector3<float> pose, float rx, float ry, int res, bool filled) {

    int res_1 = res;
    if(filled) {
        glBegin(GL_TRIANGLE_FAN);
        glVertex3f(pose.x(), pose.y(), pose.z());
        res_1++;
    } else {
        glBegin(GL_LINE_LOOP);
    }
    for (int j = 0; j < res_1; j++)   {
        float theta = 2.0f * 3.1415926f * float(j) / float(res);//get the current angle 
        float xr = rx * cosf(theta);//calculate the x component 
        float yr = ry * sinf(theta);//calculate the y component

        glVertex3f(pose.x() + xr, pose.y() + yr, pose.z());//output vertex
    }
    glEnd();
}

void drawLine(tk::common::Vector3<float> a, tk::common::Vector3<float> b, float w) {

    glLineWidth(w);
    glBegin(GL_LINES);
    glVertex3f(a.x(), a.y(), a.z());
    glVertex3f(b.x(), b.y(), b.z());
    glEnd();
}

void drawRectangle(tk::common::Vector3<float> pose, tk::common::Vector2<float> size, bool filled) {
    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;

    glPushMatrix();
    glTranslatef(pose.x(), pose.y(), pose.z());
    glScalef(size.x(), size.y(), 0.0f);
    
    glBegin(GL_QUADS);
    glVertex3f(-1,-1,0);
    glVertex3f(1,-1,0);
    glVertex3f(1,1,0);
    glVertex3f(-1,1,0);
    glEnd();
    
    glPopMatrix();

    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL ) ;
}

void drawCube(tk::common::Vector3<float> pose, tk::common::Vector3<float> size, bool filled) {
    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;

    glPushMatrix();
    glTranslatef(pose.x(), pose.y(), pose.z());
    glScalef(size.x(), size.y(), size.z());

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
drawEllipse(tk::common::Vector3<float> pose, float rot, float rx, float ry, int res, bool filled) 
{
    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_LINE ) ;

    float theta = 2.0 * M_PI/ float(res); 
    float c = cosf(theta);//precalculate the sine and cosine
    float s = sinf(theta);
    float t;

    float x = 1;//we start at angle = 0 
    float y = 0; 

    glPushMatrix();
    glTranslatef(pose.x(), pose.y(), pose.z());
    glRotatef(toDegrees(rot), 0.0f, 0.0f, 1.0f);
    glBegin(GL_LINE_LOOP); 
    for(int ii = 0; ii < res; ii++) 
    { 
        //apply radius and offset
        glVertex2f(x * rx, y * ry);//output vertex 

        //apply the rotation matrix
        t = x;
        x = c * x - s * y;
        y = s * t + c * y;
    } 
    glEnd(); 
    glPopMatrix();

    if (!filled)
        glPolygonMode ( GL_FRONT_AND_BACK, GL_FILL ) ;
}
  
}}} // namespace name
