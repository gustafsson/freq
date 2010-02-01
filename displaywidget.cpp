#include "displaywidget.h"

#include <QApplication>
#include <QTimer>
#include <QKeyEvent>

#include <list>
#include "wavelettransform.h"

#include <tvector.h>

typedef tvector<3,GLdouble> GLvector;
template<typename f>
GLvector gluProject(tvector<3,f> obj, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0) {
  GLvector win;
  bool s = (GLU_TRUE == gluProject(obj[0], obj[1], obj[2], model, proj, view, &win[0], &win[1], &win[2]));
  if(r) *r=s;
  return win;
}
template<typename f>
GLvector gluUnProject(tvector<3,f> win, const GLdouble* model, const GLdouble* proj, const GLint *view, bool *r=0) {
  GLvector obj;
  bool s = (GLU_TRUE == gluUnProject(win[0], win[1], win[2], model, proj, view, &obj[0], &obj[1], &obj[2]));
  if(r) *r=s;
  return obj;
}
template<typename f>
GLvector gluProject(tvector<3,f> obj, bool *r=0) {
  GLdouble model[16], proj[16];
  GLint view[4];
  glGetDoublev(GL_MODELVIEW_MATRIX, model);
  glGetDoublev(GL_PROJECTION_MATRIX, proj);
  glGetIntegerv(GL_VIEWPORT, view);
  return gluProject(obj, model, proj, view, r);
}
template<typename f>
GLvector gluUnProject(tvector<3,f> win, bool *r=0) {
  GLdouble model[16], proj[16];
  GLint view[4];
  glGetDoublev(GL_MODELVIEW_MATRIX, model);
  glGetDoublev(GL_PROJECTION_MATRIX, proj);
  glGetIntegerv(GL_VIEWPORT, view);
  return gluUnProject(win, model, proj, view, r);
}


using namespace std;
//using namespace Magick;

float MouseControl::deltaX( float x )
{
  if( down )
    return x - lastx;
  
  return 0;
}
float MouseControl::deltaY( float y )
{
  if( down )
    return y - lasty;
  
  return 0;
}

bool MouseControl::worldPos(GLdouble &ox, GLdouble &oy)
{
  return worldPos(this->lastx, this->lasty, ox, oy);
}
bool MouseControl::worldPos(GLdouble x, GLdouble y, GLdouble &ox, GLdouble &oy)
{
  GLdouble s;
  bool test[2];
  GLvector win_coord, world_coord[2];
  
  win_coord = GLvector(x, y, 0.1);
  
  world_coord[0] = gluUnProject<GLdouble>(win_coord, &test[0]);
  //printf("CamPos1: %f: %f: %f\n", world_coord[0][0], world_coord[0][1], world_coord[0][2]);
  
  win_coord[2] = 0.6;
  world_coord[1] = gluUnProject<GLdouble>(win_coord, &test[1]);
  //printf("CamPos2: %f: %f: %f\n", world_coord[1][0], world_coord[1][1], world_coord[1][2]);
  
  s = (-world_coord[0][1]/(world_coord[1][1]-world_coord[0][1]));
  
  ox = -s * (world_coord[1][0]-world_coord[0][0]);
  oy = s * (world_coord[1][2]-world_coord[0][2]);
  
  float minAngle = 15;
  if( s < 0 || world_coord[0][1]-world_coord[1][1] < sin(minAngle *(M_PI/180)) * (world_coord[0]-world_coord[1]).length() )
    return false;

  return test[0] && test[1];
}

void MouseControl::press( float x, float y )
{
  update( x, y );
  down = true;
}
void MouseControl::update( float x, float y )
{
  lastx = x;
  lasty = y;
}
void MouseControl::release()
{
  down = false;
}


int DisplayWidget::lastKey = 0;

DisplayWidget::DisplayWidget( boost::shared_ptr<WavelettTransform> wavelett, int timerInterval ) : QGLWidget( ),
  wavelett( wavelett ),
  px(0), py(0), pz(-10),
  rx(45), ry(225), rz(0),
  qx(0), qy(0), qz(-3.6),
  prevX(0), prevY(0)
{
    qx = 5 * wavelett->getOriginalWaveform()->_waveformData->getNumberOfElements().width / (float)wavelett->getOriginalWaveform()->_sample_rate;
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
  switch ( e->button() )
  {
    case Qt::LeftButton:
      
      leftButton.press( e->x(), this->height() - e->y() );
      //printf("LeftButton: Press\n");
      break;
      
    case Qt::MidButton:
      middleButton.press( e->x(), this->height() - e->y() );
      //printf("MidButton: Press\n");
      break;
      
    case Qt::RightButton:
    {
      rightButton.press( e->x(), this->height() - e->y() );
      //printf("RightButton: Press\n");
    }
      break;
      
    default:
      break;
  }
  glDraw();
    prevX = e->x(),
    prevY = e->y();
}

void DisplayWidget::mouseReleaseEvent ( QMouseEvent * e )
{
  switch ( e->button() )
  {
    case Qt::LeftButton:
      leftButton.release();
      //printf("LeftButton: Release\n");
      break;
      
    case Qt::MidButton:
      middleButton.release();
      //printf("MidButton: Release\n");
      break;
      
    case Qt::RightButton:
      rightButton.release();
      //printf("RightButton: Release\n");
      break;
      
    default:
      break;
  }
}

void DisplayWidget::wheelEvent ( QWheelEvent *e )
{
  float ps = 0.005;
  float rs = 0.08;
  if( e->orientation() == Qt::Horizontal )
  {
    ry -= rs * e->delta();
  }
  else
  {
    pz -= ps * e->delta();
    //rx -= ps * e->delta();
  }
  
  glDraw();
}

void DisplayWidget::mouseMoveEvent ( QMouseEvent * e )
{
  float rs = 0.2,
        ps = 0.002 - pz/1000,
        deg2rad = 3.1415926535/180;
  
  int x = e->x(), y = this->height() - e->y();
  
  //Controlling the rotation with the left button.
  ry += rs * leftButton.deltaX( x );
  rx -= rs * leftButton.deltaY( y );
  if (rx<0) rx=0;
  if (rx>90) { rx=90; orthoview=1; }
  if (0<orthoview && rx<90) { rx=90; orthoview=0; }

  //Controlling the the position with the right button.
  
  /*qx += 0.01 * ( cos(ry * deg2rad) * rightButton.deltaX( x ) 
              - sin(rx * deg2rad) * sin(ry * deg2rad) * -2*rightButton.deltaY( y ) );
  qz += 0.01 * ( sin(ry * deg2rad) * rightButton.deltaX( x ) 
              + sin(rx * deg2rad) * cos(ry * deg2rad) * -2*rightButton.deltaY( y ) );*/
  if( rightButton.isDown() )
  {
    GLvector last, current;
    if( rightButton.worldPos(last[0], last[1]) &&
        rightButton.worldPos(x, y, current[0], current[1]) )
    {
      float l = wavelett->getOriginalWaveform()->_waveformData->getNumberOfElements().width / (float)wavelett->getOriginalWaveform()->_sample_rate;

      qx += current[0] - last[0];
      qz += current[1] - last[1];

      if (qx<0) qx=0;
      if (qz>0) qz=0;
      if (qz<-8) qz=-8;
      if (qx>10*l) qx=10*l;
    }
  }
  
  
  //Updating the buttons
  leftButton.update( x, y );
  rightButton.update( x, y );
  middleButton.update( x, y );
  
  glDraw();
}

void DisplayWidget::timeOut()
{
  //printf("Timeout\n");
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

    glBegin(GL_LINE_STRIP);
            glColor3f(0,0,0);         glVertex3f( v1.x, v1.y, v1.z );
            glColor3f(1,0,0);         glVertex3f( px, py, pz );
    glEnd();

    glTranslatef( px, py, pz );

    glRotatef( rx, 1, 0, 0 );
    glRotatef( fmod(fmod(ry,360)+360, 360) * (1-orthoview) + 180*orthoview, 0, 1, 0 );
    glRotatef( rz, 0, 0, 1 );

    //drawArrows();

    //glTranslatef(-1.5f,0.0f,-6.0f);
    glTranslatef( qx, qy, qz );

    //drawColorFace();
    glScalef(-1,1-.99*orthoview,1);

    orthoview.TimeStep(.08);
//        repaint();

    glPushMatrix();
        glTranslatef( 0, 0, 6 );
        glColor3f(0,1,0);
        drawWaveform(wavelett->getInverseWaveform());

        glTranslatef( 0, 0, 2 );
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
        { 0, 0, 0 },
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
                glVertex3f( ifs*t, 0, s*data[t + c*n.width]);

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
