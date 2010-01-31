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
    yscale = Yscale_LogLinear;
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
    glScalef(-1,1,1);

    glPushMatrix();
        glTranslatef( 0, 0, 6 );
        glColor3f(0,1,0);
        drawWaveform(wavelett->getInverseWaveform());

        glTranslatef( 0, 0, 1.5 );
        glColor3f(1,0,0);
        drawWaveform(wavelett->getOriginalWaveform());
    glPopMatrix();

    drawWavelett();
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

int clamp(int val, int max) {
    if (val<0) return 0;
    if (val>max) return max;
    return val;
}

void setWavelengthColor( float wavelengthScalar ) {
    const float spectrum[][3] = {
        { 1, 0, 1 },
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 0, 1, 0 },
        { 1, 1, 0 },
        { 1, 0, 0 }};

    unsigned count = sizeof(spectrum)/sizeof(spectrum[0]);
    float f = count*wavelengthScalar;
    unsigned i = clamp(f, count-1);
    unsigned j = clamp(f+1, count-1);
    float t = f-i;

    GLfloat rgb[] = {  spectrum[i][0]*(1-t) + spectrum[j][0]*t,
                       spectrum[i][1]*(1-t) + spectrum[j][1]*t,
                       spectrum[i][2]*(1-t) + spectrum[j][2]*t
                   };
    glColor3fv( rgb );
}

void DisplayWidget::drawWaveform(boost::shared_ptr<Waveform> waveform)
{
    cudaExtent n = waveform->_waveformData->getNumberOfElements();
    //waveform->_waveformData->getCudaGlobal();
    const float* data = waveform->_waveformData->getCpuMemory();

    n.height = 1;
    float ifs = 10./waveform->_sample_rate; // step per sample
    float max = 1e-6;
    for (unsigned c=0; c<n.height; c++)
    {
        for (unsigned t=0; t<n.width; t++)
            if (fabsf(data[t + c*n.width])>max)
                max = fabsf(data[t + c*n.width]);
    }
    float s = 1/max;

    for (unsigned c=0; c<n.height; c++)
    {
        glTranslatef(0, 0, -.5); // different channels along y
        glBegin(GL_LINE_STRIP);
            // glColor3f(1-c,c,0);
            for (unsigned t=0; t<n.width; t++) {
                glVertex3f( ifs*t, s*data[t + c*n.width], 0);

                if (fabsf(data[t + c*n.width])>max)
                    max = fabsf(data[t + c*n.width]);
            }
        glEnd();
    }
}


void DisplayWidget::drawWavelett()
{
    static int drawn = 0;
    static GLuint listIndex = glGenLists(1);

    if (1<drawn) {
        glCallList(listIndex);
        return;
    }
    drawn++;

    if (1<drawn)
        glNewList(listIndex, GL_COMPILE);

    boost::shared_ptr<TransformData> transform = wavelett->getWavelettTransform();

    cudaExtent n = transform->transformData->getNumberOfElements();
    const float* data = transform->transformData->getCpuMemory();

    float ifs = 10./transform->sampleRate; // step per sample

    glTranslatef(0, 0, -(2-1)*0.5); // different channels along y

    static float max = 1;
    float s = 1/max;
    max = 0;
    int fstep = 1;
    int tstep = 10;
    float depthScale = 5.f/n.height;

    glEnable(GL_NORMALIZE);
    for (unsigned fi=0; fi+fstep<n.height; fi+=fstep)
    {
        glBegin(GL_TRIANGLE_STRIP);
            float v[3][4] = {{0}};

            int tmax = n.width>>1;
            for (int t=-1; t<=tmax; t+=tstep)
            {
                for (unsigned dt=0; dt<2; dt++)
                    for (unsigned df=0; df<4; df++)
                        v[dt][df] = v[dt+1][df];
                for (unsigned df=0; df<4; df++) {
                    float real = data[clamp(t, tmax-1)*2  + clamp(fi+(df-1)*fstep, n.height-1)*n.width];
                    float complex = data[clamp(t, tmax-1)*2+1  + clamp(fi+(df-1)*fstep, n.height-1)*n.width];

                    //float phase = atan2(complex, real);
                    float amplitude = sqrtf(real*real+complex*complex);
                    switch (yscale) {
                        case Yscale_Linear:
                            v[2][df] = amplitude;
                            break;
                        case Yscale_ExpLinear:
                            v[2][df] = amplitude * exp(.001*fi);
                            break;
                        case Yscale_LogLinear:
                            v[2][df] = amplitude;
                            v[2][df] = log(1+fabsf(v[2][df]))*(v[2][df]>0?1:-1);
                            break;
                        case Yscale_LogExpLinear:
                            v[2][df] = amplitude * exp(.001*fi);
                            v[2][df] = log(1+fabsf(v[2][df]))*(v[2][df]>0?1:-1);
                            break;
                    }
                    //v[2][df] = real;
                }

                if(0>t)
                    continue;

                setWavelengthColor( s*v[1][1] );
                float dt=(v[2][1]-v[0][1]);
                float df=(v[1][2]-v[1][0]);
                glNormal3f( -dt, 2, -df );
                glVertex3f( ifs*t, s*v[1][1], fi*depthScale);

                setWavelengthColor( s*v[1][2] );
                dt=(v[2][2]-v[0][2]);
                df=(v[1][3]-v[1][1]);
                glNormal3f( -dt, 2, -df );
                glVertex3f( ifs*t, s*v[1][2], (fi+fstep)*depthScale);

                if (fabsf(v[1][1])>max)
                    max = fabsf(v[1][1]);
            }
        glEnd();
    }
    glDisable(GL_NORMALIZE);

    if (1<drawn)
        glEndList();
}
