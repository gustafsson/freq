#ifndef TOOLS_TIMELINEVIEW_H
#define TOOLS_TIMELINEVIEW_H

#include <QGLWidget>
#include "signal/sink.h"
#include "ui/mousecontrol.h"


namespace Sawe {
    class Project;
};

namespace Tools {

class RenderView;

class TimelineView:
        public QGLWidget
{
    Q_OBJECT
public:
    TimelineView(Sawe::Project* p, RenderView* render_view);
    virtual ~TimelineView();

    // overloaded from Signal::Sink
    // TODO move into TimelineController or TimelineModel?
    //virtual void    put( Signal::pBuffer , Signal::pOperation );
    //virtual void    add_expected_samples( const Signal::Intervals& );

    void userinput_update();

protected slots:
    void getLengthNow();

protected:
    /// @overload QGLWidget::initializeGL()
    virtual void initializeGL();

    /// @overload QGLWidget::resizeGL()
    virtual void resizeGL( int width, int height );

    /// @overload QGLWidget::paintGL()
    virtual void paintGL();

    void setupCamera( bool staticTimeLine = false );

private:
    friend class TimelineController; // TODO remove

    float   _xscale,
            _xoffs,
            _barHeight,
            _length;
    int _width, _height;
    Sawe::Project* _project;
    RenderView* _render_view;
};

} // namespace Tools

#endif // TOOLS_TIMELINEVIEW_H
