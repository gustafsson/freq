#ifndef TOOLS_TIMELINEVIEW_H
#define TOOLS_TIMELINEVIEW_H

// Sonic AWE
#include "signal/sink.h"
#include "ui/mousecontrol.h"

// gpumisc
#include <gl.h>

// Qt
#include <QGLWidget>

// boost
#include <boost/scoped_ptr.hpp>

class GlFrameBuffer;

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
    int _except_count;

    boost::scoped_ptr<GlFrameBuffer> _timeline_fbo;
    boost::scoped_ptr<GlFrameBuffer> _timeline_bar_fbo;
};

} // namespace Tools

#endif // TOOLS_TIMELINEVIEW_H
