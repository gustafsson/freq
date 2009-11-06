#ifndef DISPLAYWIDGET_H
#define DISPLAYWIDGET_H

#include <QGLWidget>

class DisplayWidget : public QGLWidget
{
public:
    DisplayWidget( int timerInterval=0 );
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
  QTimer *m_timer;
  float px, py, pz,
        rx, ry, rz,
        qx, qy, qz;
  int prevX, prevY;
};

#endif // DISPLAYWIDGET_H

