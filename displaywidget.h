#ifndef DISPLAYWIDGET_H
#define DISPLAYWIDGET_H

#include <QGLWidget>
#include "wavelettransform.h"
#include <boost/shared_ptr.hpp>

class MouseControl
{
private:
  float lastx;
  float lasty;
  bool down;
  
public:
  MouseControl(): down( false ){};
  
  float deltaX( float x );
  float deltaY( float y );
  
  bool worldPos(GLdouble &ox, GLdouble &oy);
  static bool worldPos(GLdouble x, GLdouble y, GLdouble &ox, GLdouble &oy);
  
  bool isDown(){return down;};
  
  void press( float x, float y );
  void update( float x, float y );
  void release();
};

struct MyVector{
  float x, y, z;
};

class DisplayWidget : public QGLWidget
{
public:
    DisplayWidget( boost::shared_ptr<WavelettTransform> wavelett, int timerInterval=0 );
    ~DisplayWidget();
  static int lastKey;

protected:
  virtual void initializeGL();
  virtual void resizeGL( int width, int height );
  virtual void paintGL();

  virtual void mousePressEvent ( QMouseEvent * e );
  virtual void mouseReleaseEvent ( QMouseEvent * e );
  virtual void wheelEvent ( QWheelEvent *event );
  virtual void mouseMoveEvent ( QMouseEvent * e );
  virtual void timeOut();

protected slots:
  virtual void timeOutSlot();

private:
  boost::shared_ptr<WavelettTransform> wavelett;
  QTimer *m_timer;
  float px, py, pz,
        rx, ry, rz,
        qx, qy, qz;
  int prevX, prevY, targetQ;
  
  GLint viewport[4];
  GLdouble modelMatrix[16];
  GLdouble projectionMatrix[16];
  
  MyVector v1, v2;

  void drawArrows();
  void drawColorFace();
  void drawWaveform(boost::shared_ptr<Waveform> waveform);
  void drawWavelett();
  
  MouseControl leftButton;
  MouseControl rightButton;
  MouseControl middleButton;
};

#endif // DISPLAYWIDGET_H

