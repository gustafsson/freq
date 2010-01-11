#ifndef DISPLAYWIDGET_H
#define DISPLAYWIDGET_H

#include <QGLWidget>
#include "wavelettransform.h"
#include <boost/shared_ptr.hpp>

class DisplayWidget : public QGLWidget
{
public:
    DisplayWidget( boost::shared_ptr<WavelettTransform> wavelet, int timerInterval=0 );
    ~DisplayWidget();
  static int lastKey;

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
  boost::shared_ptr<Spectogram> wavelet;
  QTimer *m_timer;
  float px, py, pz,
        rx, ry, rz,
        qx, qy, qz;
  int prevX, prevY;

  void drawArrows();
  void drawColorFace();
  void drawWaveform(boost::shared_ptr<Waveform> waveform);
  void drawWavelet();
};

#endif // DISPLAYWIDGET_H

