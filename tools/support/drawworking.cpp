#include "drawworking.h"

// OpenGL
#ifndef __APPLE__
#include <GL/glut.h>
#else
#include <GLUT/glut.h>
#endif

// cos, sin, M_PI
#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <math.h>

// min, max
#include <algorithm>

namespace Tools {
    namespace Support {

static void
        drawCircleSector(float x, float y, float radius, float start, float end)
{
    int numSteps = ((end - start) * radius) * 50.0;
    float step = (end - start) / numSteps;

    glBegin(GL_TRIANGLE_FAN);
    glVertex3f( x, y, 0.0f);
    for(int i = 0; i <= numSteps; i++)
    {
        glVertex3f( x + radius * cos(start + i * step), y - radius * sin(start + i * step), 0.0f);
    }
    glEnd();
}


static void
        drawRect(float x, float y, float width, float height)
{
    glBegin(GL_QUADS);
    glVertex3f(x, y, 0.0f);
    glVertex3f(x + width, y, 0.0f);
    glVertex3f(x + width, y + height, 0.0f);
    glVertex3f(x, y + height, 0.0f);
    glEnd();
}


static void
        drawRectRing(int rects, float irad, float orad)
{
    float height = (irad / rects) * M_PI * 1.2;
    float width = orad - irad;
    float step = 360.0 / rects;

    for(int i = 0; i < rects; i++)
    {
        glPushMatrix();
        glRotatef(step * i, 0, 0, 1);
        drawRect(irad, -height/2, width, height);
        glPopMatrix();
    }
}


static void
        drawRoundRect(float width, float height, float roundness)
{
    roundness = std::max(0.01f, roundness);
    float radius = std::min(width, height) * roundness * 0.5;
    width = width - 2.0 * radius;
    height = height - 2.0 * radius;

    drawRect(-width/2.0f, -height/2.0f, width, height);
    drawRect(-(width + 2.0 * radius)/2.0f, -height/2.0f, (width + 2.0 * radius), height);
    drawRect(-width/2.0f, -(height + 2.0 * radius)/2.0f, width, (height + 2.0 * radius));

    drawCircleSector(-width/2.0f, -height/2.0f, radius, M_PI/2.0f, M_PI);
    drawCircleSector(width/2.0f, -height/2.0f, radius, 0, M_PI/2.0f);
    drawCircleSector(width/2.0f, height/2.0f, radius, -M_PI/2.0f, 0);
    drawCircleSector(-width/2.0f, height/2.0f, radius, -M_PI, -M_PI/2.0f);
}


void DrawWorking::
        drawWorking(int viewport_width, int viewport_height)
{
    static float computing_rotation = 0.0;

    glDepthFunc(GL_LEQUAL);
    glDisable(GL_DEPTH_TEST);
    glMatrixMode(GL_PROJECTION);
    glPushMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPushMatrix();

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho( viewport_width, 0, viewport_height, 0, -1, 1);

    glTranslatef( 30, 30, 0 );
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glScalef(60, 60, 1);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);

    glColor4f(1, 1, 1, 0.5);
    glPushMatrix();
    glRotatef(computing_rotation, 0, 0, 1);
    drawRectRing(15, 0.10, 0.145);
    glRotatef(-2*computing_rotation, 0, 0, 1);
    drawRectRing(20, 0.15, 0.2);
    computing_rotation += 5;
    glPopMatrix();

    glColor4f(0, 0, 1, 0.5);
    drawRoundRect(0.5, 0.5, 0.5);
    glColor4f(1, 1, 1, 0.5);
    drawRoundRect(0.55, 0.55, 0.55);

    //glDisable(GL_BLEND);
    //glDisable(GL_DEPTH_TEST);

    glMatrixMode(GL_PROJECTION);
    glPopMatrix();
    glMatrixMode(GL_MODELVIEW);
    glPopMatrix();
    glDepthFunc(GL_LEQUAL);
}

    } // namespace Support
} // namespace Tools
