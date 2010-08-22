#ifndef SAWETIMELINEWIDGET_H
#define SAWETIMELINEWIDGET_H

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
    TimelineWidget( Signal::pSink displayWidget );
    virtual ~TimelineWidget();

    // overloaded from Signal::Sink
    virtual void    put( Signal::pBuffer , Signal::pSource );
    virtual void    add_expected_samples( const Signal::SamplesIntervalDescriptor& );

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
    MouseControl moveButton;

    Signal::pSink _displayWidget;
    DisplayWidget * getDisplayWidget();
};

} // namespace Sawe

#endif // SAWECSV_H
