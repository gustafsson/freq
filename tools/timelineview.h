#ifndef TOOLS_TIMELINEVIEW_H
#define TOOLS_TIMELINEVIEW_H

#include <QGLWidget>
#include "signal/sink.h"
#include "ui/mousecontrol.h"


namespace Sawe {
    class Project;
};

namespace Tools {
class TimelineView:
        public QGLWidget
{
    Q_OBJECT
public:
    TimelineView(Sawe::Project* p, QGLWidget* displayWidget);
    virtual ~TimelineView();

    // overloaded from Signal::Sink
    // TODO move into TimelineController or TimelineModel?
    //virtual void    put( Signal::pBuffer , Signal::pOperation );
    //virtual void    add_expected_samples( const Signal::Intervals& );

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
    Ui::MouseControl moveButton;
};

} // namespace Tools

#endif // TOOLS_TIMELINEVIEW_H
