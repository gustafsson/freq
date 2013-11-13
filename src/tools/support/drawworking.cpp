#include "drawworking.h"

// gpumisc
#include "gl.h"
#include "glPushContext.h"

// std
#if defined(_MSC_VER)
#define _USE_MATH_DEFINES
#endif
#include <math.h> // cos, sin, M_PI
#include <algorithm> // std::min, std::max
#include <stdlib.h> //  error C2381: 'exit' : redefinition; __declspec(noreturn) differs

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

    glPushMatrixContext push_model( GL_MODELVIEW );

    for(int i = 0; i < rects; i++)
    {
        glRotatef(step, 0, 0, 1);
        //glPushMatrixContext push_model( GL_MODELVIEW );
        //glRotatef(step * i, 0, 0, 1);
        drawRect(irad, -height/2, width, height);
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
        drawWorking(int viewport_width, int viewport_height, bool crashed)
{
    static float computing_rotation = 0.0;

    glPushAttribContext push_attribs;

    glPushMatrixContext push_proj( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( viewport_width, 0, viewport_height, 0, -1, 1);

    // translates GL_PROJECTION to so that (0,0) is (30, 30) from the top right
    glTranslatef( 30, 30, 0 );

    glPushMatrixContext push_model( GL_MODELVIEW );

    glLoadIdentity();
    glScalef(60, 60, 1);

    glDepthFunc(GL_LESS);

    glEnable(GL_BLEND);

    glColor4f(1, 1, 1, 0.3);
    if (crashed)
        glColor4f(1, 0, 0, 0.3);

    {
        glPushMatrixContext mc(GL_MODELVIEW);

        glRotatef(computing_rotation, 0, 0, 1);
        drawRectRing(5, 0.10, 0.145);
        glRotatef(-2*computing_rotation, 0, 0, 1);
        drawRectRing(7, 0.15, 0.2);
        if (!crashed)
            computing_rotation += 5;
    }

    glColor4f(0.3, 0.3, 0.3, 0.3);
    drawRoundRect(0.5, 0.5, 0.5);
    glColor4f(1, 1, 1, 0.3);
    drawRoundRect(0.55, 0.55, 0.55);

    glDepthFunc(GL_LEQUAL);
}

    } // namespace Support
} // namespace Tools
