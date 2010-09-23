#include "selectionview.h"
#include "heightmap/renderer.h" // GLvector
#include "sawe/project.h"
#include "ui/comboboxaction.h"
#include "ui/mainwindow.h"
#include "ui_mainwindow.h"

#include <GL/gl.h>
#include <QTimer>
#include <QToolBar>

using namespace std;

namespace Tools
{

SelectionView::
        SelectionView(SelectionModel* model)
            :
            _playbackMarker(-1),
            model(model)
{
    /*ui->actionToolSelect->setEnabled( true );
    ui->actionActivateSelection->setEnabled( true );
    ui->actionSquareSelection->setEnabled( true );
    ui->actionSplineSelection->setEnabled( true );
    ui->actionPolygonSelection->setEnabled( true );
    ui->actionPeakSelection->setEnabled( true );*/
    // ui->actionPeakSelection->setChecked( false );

    Ui::SaweMainWindow* main = model->project->mainWindow();
    QToolBar* toolBarTool = new QToolBar(main);
    toolBarTool->setObjectName(QString::fromUtf8("toolBarTool"));
    toolBarTool->setEnabled(true);
    toolBarTool->setContextMenuPolicy(Qt::NoContextMenu);
    toolBarTool->setToolButtonStyle(Qt::ToolButtonIconOnly);
    main->addToolBar(Qt::TopToolBarArea, toolBarTool);

    {   Ui::ComboBoxAction * qb = new Ui::ComboBoxAction();
        qb->addActionItem( main->ui->actionActivateSelection );
        qb->addActionItem( main->ui->actionSquareSelection );
        qb->addActionItem( main->ui->actionSplineSelection );
        qb->addActionItem( main->ui->actionPolygonSelection );
        qb->addActionItem( main->ui->actionPeakSelection );

        toolBarTool->addWidget( qb );
    }
}


void SelectionView::
        drawSelection()
{
    drawSelectionCircle();
    drawPlaybackMarker();
}


void SelectionView::
        drawSelectionSquare()
{
    float l = model->project->head_source()->length();

    MyVector* selection = model->selection;

    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);
    float
    x1 = max(0.f, min(selection[0].x, selection[1].x)),
    z1 = max(0.f, min(selection[0].z, selection[1].z)),
    x2 = min(l, max(selection[0].x, selection[1].x)),
    z2 = min(1.f, max(selection[0].z, selection[1].z));
    float y = 1;


    glBegin(GL_QUADS);
    glVertex3f( 0, y, 0 );
    glVertex3f( 0, y, 1 );
    glVertex3f( x1, y, 1 );
    glVertex3f( x1, y, 0 );

    glVertex3f( x1, y, 0 );
    glVertex3f( x2, y, 0 );
    glVertex3f( x2, y, z1 );
    glVertex3f( x1, y, z1 );

    glVertex3f( x1, y, 1 );
    glVertex3f( x2, y, 1 );
    glVertex3f( x2, y, z2 );
    glVertex3f( x1, y, z2 );

    glVertex3f( l, y, 0 );
    glVertex3f( l, y, 1 );
    glVertex3f( x2, y, 1 );
    glVertex3f( x2, y, 0 );


    if (x1>0) {
        glVertex3f( x1, y, z1 );
        glVertex3f( x1, 0, z1 );
        glVertex3f( x1, 0, z2 );
        glVertex3f( x1, y, z2 );
        glVertex3f( 0, y, 0 );
        glVertex3f( 0, 0, 0 );
        glVertex3f( 0, 0, 1 );
        glVertex3f( 0, y, 1 );
    } else {
        glVertex3f( 0, y, 0 );
        glVertex3f( 0, 0, 0 );
        glVertex3f( 0, 0, z1 );
        glVertex3f( 0, y, z1 );
        glVertex3f( 0, y, z2 );
        glVertex3f( 0, 0, z2 );
        glVertex3f( 0, 0, 1 );
        glVertex3f( 0, y, 1 );
    }

    if (x2<l) {
        glVertex3f( x2, y, z1 );
        glVertex3f( x2, 0, z1 );
        glVertex3f( x2, 0, z2 );
        glVertex3f( x2, y, z2 );
        glVertex3f( l, y, 0 );
        glVertex3f( l, 0, 0 );
        glVertex3f( l, 0, 1 );
        glVertex3f( l, y, 1 );
    } else {
        glVertex3f( l, y, 0 );
        glVertex3f( l, 0, 0 );
        glVertex3f( l, 0, z1 );
        glVertex3f( l, y, z1 );
        glVertex3f( l, y, z2 );
        glVertex3f( l, 0, z2 );
        glVertex3f( l, 0, 1 );
        glVertex3f( l, y, 1 );
    }

    if (z1>0) {
        glVertex3f( x1, y, z1 );
        glVertex3f( x1, 0, z1 );
        glVertex3f( x2, 0, z1 );
        glVertex3f( x2, y, z1 );
        glVertex3f( 0, y, 0 );
        glVertex3f( 0, 0, 0 );
        glVertex3f( l, 0, 0 );
        glVertex3f( l, y, 0 );
    } else {
        glVertex3f( 0, y, 0 );
        glVertex3f( 0, 0, 0 );
        glVertex3f( x1, 0, 0 );
        glVertex3f( x1, y, 0 );
        glVertex3f( x2, y, 0 );
        glVertex3f( x2, 0, 0 );
        glVertex3f( l, 0, 0 );
        glVertex3f( l, y, 0 );
    }

    if (z2<1) {
        glVertex3f( x1, y, z2 );
        glVertex3f( x1, 0, z2 );
        glVertex3f( x2, 0, z2 );
        glVertex3f( x2, y, z2 );
        glVertex3f( 0, y, 1 );
        glVertex3f( 0, 0, 1 );
        glVertex3f( l, 0, 1 );
        glVertex3f( l, y, 1 );
    } else {
        glVertex3f( 0, y, 1 );
        glVertex3f( 0, 0, 1 );
        glVertex3f( x1, 0, 1 );
        glVertex3f( x1, y, 1 );
        glVertex3f( x2, y, 1 );
        glVertex3f( x2, 0, 1 );
        glVertex3f( l, 0, 1 );
        glVertex3f( l, y, 1 );
    }
    glEnd();
    //glDisable(GL_BLEND);
    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_QUADS);
    if (x1>0) {
        glVertex3f( x1, y, z1 );
        glVertex3f( x1, 0, z1 );
        glVertex3f( x1, 0, z2 );
        glVertex3f( x1, y, z2 );
    }

    if (x2<l) {
        glVertex3f( x2, y, z1 );
        glVertex3f( x2, 0, z1 );
        glVertex3f( x2, 0, z2 );
        glVertex3f( x2, y, z2 );
    }

    if (z1>0) {
        glVertex3f( x1, y, z1 );
        glVertex3f( x1, 0, z1 );
        glVertex3f( x2, 0, z1 );
        glVertex3f( x2, y, z1 );
    }

    if (z2<1) {
        glVertex3f( x1, y, z2 );
        glVertex3f( x1, 0, z2 );
        glVertex3f( x2, 0, z2 );
        glVertex3f( x2, y, z2 );
    }
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
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
        drawSelectionCircle()
{
    MyVector* selection = model->selection;
    float
        x = selection[0].x,
        z = selection[0].z,
        _rx = fabs(selection[1].x-selection[0].x),
        _rz = fabs(selection[1].z-selection[0].z);
    float y = 1;

    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);
    glBegin(GL_TRIANGLE_STRIP);
    for (unsigned k=0; k<=360; k++) {
        float s = z + _rz*sin(k*M_PI/180);
        float c = x + _rx*cos(k*M_PI/180);
        glVertex3f( c, 0, s );
        glVertex3f( c, y, s );
    }
    glEnd();

    glLineWidth(3.2f);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_LINE_LOOP);
    for (unsigned k=0; k<360; k++) {
        float s = z + _rz*sin(k*M_PI/180);
        float c = x + _rx*cos(k*M_PI/180);
        glVertex3f( c, y, s );
    }
    glEnd();
    glLineWidth(0.5f);
    glDepthMask(true);
    //glDisable(GL_BLEND);
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
        drawPlaybackMarker()
{
    if (0>_playbackMarker)
        return;

    //glEnable(GL_BLEND);
    glDepthMask(false);
    //glBlendFunc(GL_ONE, GL_ONE_MINUS_SRC_ALPHA);
    glColor4f( 0, 0, 0, .5);

    MyVector* selection = model->selection;

    float
        t = _playbackMarker,
        x = selection[0].x,
        y = 1,
        z = selection[0].z,
        _rx = selection[1].x-selection[0].x,
        _rz = selection[1].z-selection[0].z,
        z1 = z-sqrtf(1 - (x-t)*(x-t)/_rx/_rx)*_rz,
        z2 = z+sqrtf(1 - (x-t)*(x-t)/_rx/_rx)*_rz;


    glBegin(GL_QUADS);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z2 );
        glVertex3f( t, y, z1 );
    glEnd();

    //glDisable(GL_BLEND);
    glDepthMask(true);

    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
    glPolygonOffset(1.f, 1.f);
    glBegin(GL_QUADS);
        glVertex3f( t, 0, z1 );
        glVertex3f( t, 0, z2 );
        glVertex3f( t, y, z2 );
        glVertex3f( t, y, z1 );
    glEnd();
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    // TODO make sure to repaint the area as the selection marker move
    // with time
    QTimer::singleShot(10, model->project->mainWindow(), SLOT(update()));
}

} // namespace Tools
