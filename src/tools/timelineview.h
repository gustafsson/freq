#ifndef TOOLS_TIMELINEVIEW_H
#define TOOLS_TIMELINEVIEW_H

// Sonic AWE
#include "signal/sink.h"
#include "ui/mousecontrol.h"
#include "heightmap/position.h"

// gpumisc
#include "gl.h"

// Qt
#include <QGLWidget>
#include <QBoxLayout>

// boost
#include <boost/scoped_ptr.hpp>
#include <boost/date_time/posix_time/ptime.hpp>

class GlFrameBuffer;

namespace Sawe {
    class Project;
};


namespace Tools {

namespace Support { class ToolSelector; }

class RenderView;

class TimelineView:
        public QGLWidget
{
    Q_OBJECT
public:
    TimelineView(Sawe::Project* p, RenderView* render_view);
    virtual ~TimelineView();

    Heightmap::Position getSpacePos( QPointF pos, bool* success = 0 );

    void redraw();

    Support::ToolSelector* tool_selector;

signals:
    void hideMe();
    void painting();

public slots:
    void paintInGraphicsView();
    void layoutChanged( QBoxLayout::Direction direction );

protected:
    /// @overload QGLWidget::initializeGL()
    virtual void initializeGL();
    void initializeTimeline();

    /// @overload QGLWidget::resizeGL()
    void resizeGL( int x, int y, int width, int height );
    virtual void resizeGL( int width, int height );

    /// @overload QGLWidget::paintGL()
    virtual void paintGL();

    /// @overload QGLWidget::paintEvent ()
    virtual void paintEvent ( QPaintEvent * event );

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

    GLvector::T modelview_matrix[16], projection_matrix[16];
    int viewport_matrix[4];

    boost::scoped_ptr<GlFrameBuffer> _timeline_fbo;
    boost::scoped_ptr<GlFrameBuffer> _timeline_bar_fbo;
    bool _vertical;
};

} // namespace Tools

#endif // TOOLS_TIMELINEVIEW_H
