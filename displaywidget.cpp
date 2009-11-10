#include "displaywidget.h"

#include <QApplication>
#include <QTimer>
#include <QKeyEvent>
//#include <Magick++.h>

#include <list>
#include "wavelettransform.h"

using namespace std;
//using namespace Magick;

int DisplayWidget::lastKey = 0;

DisplayWidget::DisplayWidget( boost::shared_ptr<WavelettTransform> wavelett, int timerInterval ) : QGLWidget( ),
  wavelett( wavelett ),
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

    GLfloat LightAmbient[]= { 0.5f, 0.5f, 0.5f, 1.0f };
    GLfloat LightDiffuse[]= { 1.0f, 1.0f, 1.0f, 1.0f };
    GLfloat LightPosition[]= { 0.0f, 0.0f, 2.0f, 1.0f };
    glShadeModel(GL_SMOOTH);
    glLightfv(GL_LIGHT1, GL_AMBIENT, LightAmbient);
    glLightfv(GL_LIGHT1, GL_DIFFUSE, LightDiffuse);
    glLightfv(GL_LIGHT1, GL_POSITION,LightPosition);
    glEnable(GL_LIGHT1);
    glEnable(GL_LIGHTING);
    glEnable(GL_COLOR_MATERIAL);
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

    drawArrows();

    //glTranslatef(-1.5f,0.0f,-6.0f);
    glTranslatef( qx, qy, qz );

    //drawColorFace();
    //drawWaveform(wavelett->getOriginalWaveform());
    //drawWavelett();
    drawWaveform(wavelett->getInverseWaveform());
}

void DisplayWidget::drawArrows()
{
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
}

void DisplayWidget::drawColorFace()
{
    glBegin(GL_TRIANGLE_FAN);
            glColor3f( 0, 0, 0);    glVertex3f( 0.0f, 0.0f, 0.0f);
            glColor3f( 1, 0, 0);    glVertex3f( -1.0f, -1.0f, 0.0f);
            glColor3f( 0, 1, 0);    glVertex3f(1.0f,-1.0f, 0.0f);
            glColor3f( 0, 0, 1);    glVertex3f( 1.0f,1.0f, 0.0f);
            glColor3f( .7, .7, 0);    glVertex3f( -1.0f,1.0f, 0.0f);
            glColor3f( 1, 0, 0);    glVertex3f( -1.0f, -1.0f, 0.0f);
    glEnd();
}

void DisplayWidget::drawWaveform(boost::shared_ptr<Waveform> waveform)
{
    cudaExtent n = waveform->_waveformData->getNumberOfElements();
    waveform->_waveformData->getCudaGlobal();
    const float* data = waveform->_waveformData->getCpuMemory();

    float ifs = 10./waveform->_sample_rate; // step per sample

    glTranslatef(0, 0, -(2-1)*0.5); // different channels along y
    glTranslatef(-ifs*n.width/2, 0, 0); // start
    for (unsigned c=0; c<n.height; c++)
    {
        glBegin(GL_LINE_STRIP);
            glColor3f(1-c,c,0);
            for (unsigned t=0; t<n.width; t++) {
                glVertex3f( ifs*t, 10*data[t + c*n.width], c);
            }
        glEnd();
    }
}

int clamp(int val, int max) {
    if (val<0) return 0;
    if (val>max) return max;
    return val;
}

void DisplayWidget::drawWavelett()
{
    boost::shared_ptr<TransformData> transform = wavelett->getWavelettTransform();
    cudaExtent n = transform->transformData->getNumberOfElements();
    const float* data = transform->transformData->getCpuMemory();

    float ifs = 10./transform->sampleRate; // step per sample

    glTranslatef(0, 0, -(2-1)*0.5); // different channels along y
    glTranslatef(-ifs*n.width/2, 0, 0); // start

    static float max = 1;
    float s = 1/max;
    max = 0;
    int fstep = 1;
    int tstep = 400;
    float depthScale = 5.f/n.height;

    glEnable(GL_NORMALIZE);
    for (int fi=0; fi+fstep<n.height; fi+=fstep)
    {
        glBegin(GL_TRIANGLE_STRIP);
            float v[3][4] = {{0}};

            for (unsigned df=0; df<4; df++) {
                v[2][df] = data[clamp(0, n.width-1)  + clamp(fi+(df-1)*fstep, n.height-1)*n.width];
                v[2][df] = log(1+fabsf(v[2][df]))*(v[2][df]>0?1:-1);
            }

            for (unsigned t=0; t<=n.width; t+=tstep)
            {
                for (unsigned dt=0; dt<2; dt++)
                    for (unsigned df=0; df<4; df++)
                        v[dt][df] = v[dt+1][df];
                for (unsigned df=0; df<4; df++) {
                    v[2][df] = data[clamp(t, n.width-1)  + clamp(fi+(df-1)*fstep, n.height-1)*n.width];
                    v[2][df] = log(1+fabsf(v[2][df]))*(v[2][df]>0?1:-1);
                }

                glColor3f(1-s*v[1][1],s*v[1][1],0);
                float dt=(v[2][1]-v[0][1]);
                float df=(v[1][2]-v[1][0]);
                glNormal3f( dt, -2, df );
                glVertex3f( ifs*t, s*v[1][1], fi*depthScale);

                glColor3f(1-s*v[1][2],s*v[1][2],0);
                dt=(v[2][2]-v[0][2]);
                df=(v[1][3]-v[1][1]);
                glNormal3f( dt, -2, df );
                glVertex3f( ifs*t, s*v[1][2], (fi+fstep)*depthScale);

                if (fabsf(v[1][1])>max)
                    max = fabsf(v[1][1]);
            }
        glEnd();
    }
    glDisable(GL_NORMALIZE);
}
