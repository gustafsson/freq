#if 0
#include "selectionview.h"

#include "selectionmodel.h"
#include "heightmap/renderer.h" // GLvector
#include "sawe/project.h"

#include <GL/gl.h>
#include <QTimer>
#include <QToolBar>

using namespace std;

namespace Tools
{

SelectionView::
        SelectionView(SelectionModel* model)
            :
            model(model)
{

}


SelectionView::
        ~SelectionView()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


void SelectionView::
        drawSelection()
{
    drawSelectionCircle2();
}


bool SelectionView::
        insideCircle( float x1, float z1 )
{
    MyVector* selection = model->selection;
    float
        x = selection[0].x,
        z = selection[0].z,
        _rx = selection[1].x,
        _rz = selection[1].z;
    return (x-x1)*(x-x1)/_rx/_rx + (z-z1)*(z-z1)/_rz/_rz < 1;
}


void SelectionView::
        drawSelectionCircle2()
{
    float l = model->project->worker.source()->length();
    glDepthMask(false);
    glColor4f( 0, 0, 0, .5);

    MyVector* selection = model->selection;
    float
        x = selection[0].x,
        z = selection[0].z,
        _rx = fabs(selection[1].x-selection[0].x),
        _rz = fabs(selection[1].z-selection[0].z);
    float y = 1;

    // compute points in each quadrant, upper right
    std::vector<GLvector> pts[4];
    GLvector corner[4];
    corner[0] = GLvector(l,0,1);
    corner[1] = GLvector(0,0,1);
    corner[2] = GLvector(0,0,0);
    corner[3] = GLvector(l,0,0);

    for (unsigned k,j=0; j<4; j++)
    {
        bool addedLast=false;
        for (k=0; k<=90; k++)
        {
            float s = z + _rz*sin((k+j*90)*M_PI/180);
            float c = x + _rx*cos((k+j*90)*M_PI/180);

            if (s>0 && s<1 && c>0&&c<l)
            {
                if (pts[j].empty() && k>0)
                {
                    if (0==j) pts[j].push_back(GLvector( l, 0, z + _rz*sin(acos((l-x)/_rx))));
                    if (1==j) pts[j].push_back(GLvector( x + _rx*cos(asin((1-z)/_rz)), 0, 1));
                    if (2==j) pts[j].push_back(GLvector( 0, 0, z + _rz*sin(acos((0-x)/_rx))));
                    if (3==j) pts[j].push_back(GLvector( x + _rx*cos(asin((0-z)/_rz)), 0, 0));
                }
                pts[j].push_back(GLvector( c, 0, s));
                addedLast = 90==k;
            }
        }

        if (!addedLast) {
            if (0==j) pts[j].push_back(GLvector( x + _rx*cos(asin((1-z)/_rz)), 0, 1));
            if (1==j) pts[j].push_back(GLvector( 0, 0, z + _rz*sin(acos((0-x)/_rx))));
            if (2==j) pts[j].push_back(GLvector( x + _rx*cos(asin((0-z)/_rz)), 0, 0));
            if (3==j) pts[j].push_back(GLvector( l, 0, z + _rz*sin(acos((l-x)/_rx))));
        }
    }

    for (unsigned j=0; j<4; j++) {
        glBegin(GL_TRIANGLE_STRIP);
        for (unsigned k=0; k<pts[j].size(); k++) {
            glVertex3f( pts[j][k][0], 0, pts[j][k][2] );
            glVertex3f( pts[j][k][0], y, pts[j][k][2] );
        }
        glEnd();
    }


    for (unsigned j=0; j<4; j++) {
        if ( !insideCircle(corner[j][0], corner[j][2]) )
        {
            glBegin(GL_TRIANGLE_FAN);
            GLvector middle1( 0==j?l:2==j?0:corner[j][0], 0, 1==j?1:3==j?0:corner[j][2]);
            GLvector middle2( 3==j?l:1==j?0:corner[j][0], 0, 0==j?1:2==j?0:corner[j][2]);
            if ( !insideCircle(middle1[0], middle1[2]) )
                glVertex3f( middle1[0], y, middle1[2] );
            for (unsigned k=0; k<pts[j].size(); k++) {
                glVertex3f( pts[j][k][0], y, pts[j][k][2] );
            }
            if ( !insideCircle(middle2[0], middle2[2]) )
                glVertex3f( middle2[0], y, middle2[2] );
            glEnd();
        }
    }
    for (unsigned j=0; j<4; j++) {
        bool b1 = insideCircle(corner[j][0], corner[j][2]);
        bool b2 = insideCircle(0==j?l:2==j?0:corner[j][0], 1==j?1:3==j?0:corner[j][2]);
        bool b3 = insideCircle(corner[(j+1)%4][0], corner[(j+1)%4][2]);
        glBegin(GL_TRIANGLE_STRIP);
        if ( b1 )
        {
            glVertex3f( corner[j][0], 0, corner[j][2] );
            glVertex3f( corner[j][0], y, corner[j][2] );
            if ( !b2 && pts[(j+1)%4].size()>0 ) {
                glVertex3f( pts[j].back()[0], 0, pts[j].back()[2] );
                glVertex3f( pts[j].back()[0], y, pts[j].back()[2] );
            }
        }
        if ( b3 ) {
            if ( !b2 && pts[(j+1)%4].size()>1) {
                glVertex3f( pts[(j+1)%4][1][0], 0, pts[(j+1)%4][1][2] );
                glVertex3f( pts[(j+1)%4][1][0], y, pts[(j+1)%4][1][2] );
            }
            glVertex3f( corner[(j+1)%4][0], 0, corner[(j+1)%4][2] );
            glVertex3f( corner[(j+1)%4][0], y, corner[(j+1)%4][2] );
        }
        glEnd();
    }
    //glDisable(GL_BLEND);
    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonOffset(1.f, 1.f);

    for (unsigned j=0; j<4; j++) {
        glBegin(GL_LINE_STIPPLE);
        for (unsigned k=0; k<pts[j].size(); k++) {
            glVertex3f( pts[j][k][0], y, pts[j][k][2] );
        }
        glEnd();
    }

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}


void SelectionView::
        draw()
{
    // TODO render differently
    if (enabled)
        drawSelection();
    else
        drawSelection();
}


} // namespace Tools
#endif
