#ifndef DISPLAYWIDGET_H
#define DISPLAYWIDGET_H

#include <QGLWidget>
#include "wavelettransform.h"
#include <boost/shared_ptr.hpp>

class DisplayWidget : public QGLWidget
{
public:
    DisplayWidget( boost::shared_ptr<WavelettTransform> wavelett, int timerInterval=0 );
    ~DisplayWidget();
  static int lastKey;

  enum Yscale {
      Yscale_Linear,
      Yscale_ExpLinear,
      Yscale_LogLinear,
      Yscale_LogExpLinear
  } yscale;

protected:
  virtual void initializeGL();
  virtual void resizeGL( int width, int height );
  virtual void paintGL();

  virtual void mousePressEvent ( QMouseEvent * e );
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
};

#endif // DISPLAYWIDGET_H

