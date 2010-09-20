#ifndef UI_TIMELINEWIDGET_H
#define UI_TIMELINEWIDGET_H

#include <QGLWidget>
#include "signal/sink.h"
#include "ui/displaywidget.h"

namespace Ui {

class TimelineWidget:
        public QGLWidget
{
    Q_OBJECT
public:
    TimelineWidget(Sawe::Project* p, QGLWidget* displayWidget);
    virtual ~TimelineWidget();

    // overloaded from Signal::Sink
    virtual void    put( Signal::pBuffer , Signal::pOperation );
    virtual void    add_expected_samples( const Signal::Intervals& );

protected:
    // overloaded from QGLWidget
    virtual void initializeGL();
    virtual void resizeGL( int width, int height );
    virtual void paintGL();
    void setupCamera( bool staticTimeLine = false );

    // overloaded from QWidget
    virtual void wheelEvent ( QWheelEvent *e );
    virtual void mousePressEvent ( QMouseEvent * e );
    virtual void mouseReleaseEvent ( QMouseEvent * e );
    virtual void mouseMoveEvent ( QMouseEvent * e );
private:
    float   _xscale,
            _xoffs,
            _barHeight;
    int _movingTimeline;
    Sawe::Project* _project;
    MouseControl moveButton;
};

} // namespace Ui

#endif // UI_TIMELINEWIDGET_H
