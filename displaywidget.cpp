#include "displaywidget.h"

#include <QApplication>
#include <QTimer>
#include <QKeyEvent>
//#include <Magick++.h>

#include <list>

using namespace std;
//using namespace Magick;

int DisplayWidget::lastKey = 0;

DisplayWidget::DisplayWidget( int timerInterval ) : QGLWidget( ),
  px(0), py(0), pz(0),
  rx(0), ry(0), rz(0),
  qx(0), qy(0), qz(0),
  prevX(0), prevY(0)

{
    timeOut();

    if( timerInterval == 0 )
        m_timer = 0;
    else
    {
        m_timer = new QTimer( this );
        connect( m_timer, SIGNAL(timeout()), this, SLOT(timeOutSlot()) );
        m_timer->start( timerInterval );
    }
}

DisplayWidget::~DisplayWidget()
{}


void DisplayWidget::mousePressEvent ( QMouseEvent * e )
{
    prevX = e->x(),
    prevY = e->y();
}
void DisplayWidget::mouseMoveEvent ( QMouseEvent * e )
{
    float rs = 0.1,
          ps = 0.002;

    int dx = e->x() - prevX,
        dy = e->y() - prevY,
        d = dx-dy;
    prevX = e->x(),
    prevY = e->y();

    switch( lastKey ) {
        case 'A': px += d*ps; break;
        case 'S': py += d*ps; break;
        case 'D': pz += d*ps; break;
        case 'Q': rx += d*rs; break;
        case 'W': ry += d*rs; break;
        case 'E': rz += d*rs; break;
        case 'Z': qx += d*ps; break;
        case 'X': qy += d*ps; break;
        case 'C': qz += d*ps; break;
        default:
            ry += dx*rs;
            rx += dy*rs;
            break;
    }

    glDraw();
}

void DisplayWidget::timeOut()
{
    try{
} catch (...) {
    string x32= "blaj";
}
}

void DisplayWidget::timeOutSlot()
{
        timeOut();
}

void DisplayWidget::initializeGL()
{
    glShadeModel(GL_SMOOTH);

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClearDepth(1.0f);

    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_LEQUAL);
//    glDepthFunc(GL_NEVER);

    glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

}

void DisplayWidget::resizeGL( int width, int height ) {
    height = height?height:1;

    glViewport( 0, 0, (GLint)width, (GLint)height );

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluPerspective(45.0f,(GLfloat)width/(GLfloat)height,0.1f,100.0f);

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
}

void DisplayWidget::paintGL()
{
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    glLoadIdentity();

    glTranslatef( 0, 0, -3 );

    glBegin(GL_LINE_STRIP);
            glColor3f(0,0,0);         glVertex3f( 0, 0, 0 );
            glColor3f(1,0,0);         glVertex3f( px, py, pz );
    glEnd();

    glTranslatef( px, py, pz );

    glRotatef( rx, 1, 0, 0 );
    glRotatef( ry, 0, 1, 0 );
    glRotatef( rz, 0, 0, 1 );

    glBegin(GL_LINE_STRIP);
            glColor3f(1,0,0);
            glVertex3f( 0.0f, 0.0f, 0.0f);
            glVertex3f( 1.0f, 0.0f, 0.0f);
            glVertex3f( 0.9f, 0.1f, 0.0f);
            glVertex3f( 0.9f, -0.1f, 0.0f);
            glVertex3f( 1.0f, 0.0f, 0.0f);
            glVertex3f( 0.9f, 0.0f, 0.1f);
            glVertex3f( 0.9f, 0.0f, -0.1f);
            glVertex3f( 1.0f, 0.0f, 0.0f);
            glVertex3f( 0.0f, 0.0f, 0.0f);
            glColor3f(0,1,0);
            glVertex3f( 0.0f, 1.0f, 0.0f);
            glVertex3f( 0.0f, 0.9f, 0.1f);
            glVertex3f( 0.0f, 0.9f, -0.1f);
            glVertex3f( 0.0f, 1.0f, 0.0f);
            glVertex3f( 0.1f, 0.9f, 0.0f);
            glVertex3f( -0.1f, 0.9f, 0.0f);
            glVertex3f( 0.0f, 1.0f, 0.0f);
            glVertex3f( 0.0f, 0.0f, 0.0f);
            glColor3f(0,0,1);
            glVertex3f( 0.0f, 0.0f, 1.0f);
            glVertex3f( 0.0f, 0.1f, 0.9f);
            glVertex3f( 0.0f, -0.1f, 0.9f);
            glVertex3f( 0.0f, 0.0f, 1.0f);
            glVertex3f( 0.1f, 0.0f, 0.9f);
            glVertex3f( -0.1f, 0.0f, 0.9f);
            glVertex3f( 0.0f, 0.0f, 1.0f);
            glVertex3f( 0.0f, 0.0f, 0.0f);
            glColor3f(1,1,1);
            glVertex3f( qx, qy, qz );
    glEnd();

    //glTranslatef(-1.5f,0.0f,-6.0f);
    glTranslatef( qx, qy, qz );

    glBegin(GL_TRIANGLE_FAN);
            glColor3f( 0, 0, 0);    glVertex3f( 0.0f, 0.0f, 0.0f);
            glColor3f( 1, 0, 0);    glVertex3f( -1.0f, -1.0f, 0.0f);
            glColor3f( 0, 1, 0);    glVertex3f(1.0f,-1.0f, 0.0f);
            glColor3f( 0, 0, 1);    glVertex3f( 1.0f,1.0f, 0.0f);
            glColor3f( .7, .7, 0);    glVertex3f( -1.0f,1.0f, 0.0f);
            glColor3f( 1, 0, 0);    glVertex3f( -1.0f, -1.0f, 0.0f);
    glEnd();
}
