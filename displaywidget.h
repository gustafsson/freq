#ifndef DISPLAYWIDGET_H
#define DISPLAYWIDGET_H

#include <QGLWidget>
#include "wavelettransform.h"
#include <boost/shared_ptr.hpp>

class MouseControl
{
private:
  int lastx;
  int lasty;
  bool down;
  
public:
  MouseControl(): down( false ){};
  
  int deltaX( int x );
  int deltaY( int y );
  
  void press( int x, int y );
  void update( int x, int y );
  void release();
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
  int prevX, prevY;

  void drawArrows();
  void drawColorFace();
  void drawWaveform(boost::shared_ptr<Waveform> waveform);
  void drawWavelett();
  
  MouseControl leftButton;
  MouseControl rightButton;
  MouseControl middleButton;
};

#endif // DISPLAYWIDGET_H

