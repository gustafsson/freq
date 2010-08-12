#include "sawe-brushtool.h"
#include <QTabletEvent>
#define M_PI 3.1415926535

namespace Sawe {

void drawCircle(float z, float x, float rz, float rx, bool lid = true, bool line = false, bool body = true)
{
    float y = 1;

    if(body)
    {
        glBegin(GL_TRIANGLE_STRIP);
        for (unsigned k=0; k<=360; k++) {
            float s = z + rz*sin(k*M_PI/180);
            float c = x + rx*cos(k*M_PI/180);
            glVertex3f( c, 0, s );
            glVertex3f( c, y, s );
        }
        glEnd();
    }
    
    if(lid)
    {
        glBegin(GL_TRIANGLE_FAN);
	    glVertex3f( x, y, z);
        for (unsigned k=0; k<=360; k++)
        {
            float s = z + rz*sin(k*M_PI/180);
            float c = x + rx*cos(k*M_PI/180);
         	glVertex3f( c, y, s);
        }
        glEnd();
    }
    
    if(line)
    {
        glBegin(GL_LINE_LOOP);
        for (unsigned k=0; k<360; k++) {
            float s = z + rz*sin(k*M_PI/180);
            float c = x + rx*cos(k*M_PI/180);
            glVertex3f( c, y, s );
        }
        glEnd();
        glLineWidth(1.0f);
    }
    
}

void drawBlock(float z1, float x1, float s1, float z2, float x2, float s2)
{
    float xdir[2], length;
    
    xdir[1] = x2 - x1;
    xdir[0] = z1 - z2;
    length = sqrt(xdir[1] * xdir[1] + xdir[0] * xdir[0]);
    xdir[0] /= length;
    xdir[1] /= length;

    /*glBegin(GL_QUADS);
    glVertex3f(x1 + xdir[0] * s1, 0.0, z1 + xdir[1] * s1);
    glVertex3f(x1 + xdir[0] * s1, 1.0, z1 + xdir[1] * s1);
    glVertex3f(x2 + xdir[0] * s2, 1.0, z2 + xdir[1] * s2);
    glVertex3f(x2 + xdir[0] * s2, 0.0, z2 + xdir[1] * s2);
    glVertex3f(x1 - xdir[0] * s1, 0.0, z1 - xdir[1] * s1);
    glVertex3f(x1 - xdir[0] * s1, 1.0, z1 - xdir[1] * s1);
    glVertex3f(x2 - xdir[0] * s2, 1.0, z2 - xdir[1] * s2);
    glVertex3f(x2 - xdir[0] * s2, 0.0, z2 - xdir[1] * s2);
    glEnd();*/
    
    
    glLineWidth(1.0f);
    glBegin(GL_LINES);
    glVertex3f(x1 + xdir[0] * s1, 1.0, z1 + xdir[1] * s1);
    glVertex3f(x2 + xdir[0] * s2, 1.0, z2 + xdir[1] * s2);
    glVertex3f(x1 - xdir[0] * s1, 1.0, z1 - xdir[1] * s1);
    glVertex3f(x2 - xdir[0] * s2, 1.0, z2 - xdir[1] * s2);
    glEnd();
    glLineWidth(0.5f);
}

BrushTool::BrushTool(DisplayWidget *dw): BasicTool(dw)
{
    isPainting = false;
}
    
void BrushTool::render()
{
    int size = stroke.size();
    
    glDisable(GL_CULL_FACE);
    glColor4f( 0, 0, 0, 1);
    for(int i = 0; i < size - 1; i++)
    {
        drawBlock(stroke[i].y, stroke[i].x, stroke[i].size * 0.1, stroke[i + 1].y, stroke[i + 1].x, stroke[i + 1].size * 0.1);
    }
    
    /*glEnable(GL_CULL_FACE);
    glFrontFace(GL_CCW);
    glColor4f( 0, 0, 0, 0.5);
    for(int i = 0; i < size; i++)
    {
        drawCircle(stroke[i].y, stroke[i].x, stroke[i].size * 0.1, stroke[i].size * 0.1, false);
    }
    
    glEnable(GL_CULL_FACE);
    glFrontFace(GL_CW);
    for(int i = 0; i < size; i++)
    {
        drawCircle(stroke[i].y, stroke[i].x, stroke[i].size * 0.1, stroke[i].size * 0.1, false, true);
    }*/
    /*glColor4f( 0, 0, 0, 0.0);
    glDepthMask(true);
    for(int i = 0; i < size; i++)
    {
        drawCircle(stroke[i].y, stroke[i].x, stroke[i].size * 0.1, stroke[i].size * 0.1);
    }
    glColor4f( 0, 0, 0, 0.5);
    for(int i = 0; i < size; i++)
    {
        drawCircle(stroke[i].y, stroke[i].x, stroke[i].size * 0.1, stroke[i].size * 0.1, false, true);
    }*/
    glDepthMask(true);
}
QWidget *BrushTool::getSettingsWidget()
{
}

void BrushTool::tabletEvent(QTabletEvent *event)
{
    if(event->pressure() > 0.0001)
    {
        if(!isPainting) stroke.clear();
        isPainting = true;
        double x, y;
        displayWidget->worldPos((GLdouble)event->x(), (GLdouble)(displayWidget->height() - event->y()), x, y, 1.0);
        printf("tablet: pressure: %f x: %f y: %f\n", event->pressure(), x, y);
        stroke.push_back(BrushPoint(x, y, event->pressure()));
        displayWidget->update();
        //return true;
    }
    else
    {
        isPainting = false;
        //return false;
    }
}
void BrushTool::mousePressEvent(QMouseEvent * e)
{
    //return false;
    isPainting = true;
    stroke.clear();
    displayWidget->update();
    //return true;
}
void BrushTool::mouseMoveEvent(QMouseEvent * e)
{
    double x, y;
    displayWidget->worldPos((GLdouble)e->x(), (GLdouble)(displayWidget->height() - e->y()), x, y, 1.0);
    stroke.push_back(BrushPoint(x, y, 1.0));
    displayWidget->update();
    //return true;
}

void BrushTool::mouseReleaseEvent(QMouseEvent * e){isPainting = false;}

};