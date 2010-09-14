#ifndef SAWETIMELINEWIDGET_H
#define SAWETIMELINEWIDGET_H

// TODO move to saweui

#include <QGLWidget>
#include "signal/sink.h"
#include "saweui/displaywidget.h"

namespace Sawe {

class TimelineWidget:
        public QGLWidget,
        public Signal::Sink
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

} // namespace Sawe

#endif // SAWECSV_H
