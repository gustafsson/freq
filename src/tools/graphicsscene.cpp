#include "graphicsscene.h"
#include "glPushContext.h"
#include "GlException.h"
#include "tools/support/toolselector.h"
#include "tools/support/renderviewinfo.h"

#include <QTimer>
#include <QGraphicsSceneMouseEvent>

//#define TIME_PAINTGL
#define TIME_PAINTGL if(0)

//#define DEBUG_EVENTS
#define DEBUG_EVENTS if(0)

//#define TIME_PAINTGL_DETAILS
#define TIME_PAINTGL_DETAILS if(0)

namespace Tools {

GraphicsScene::GraphicsScene(RenderView* renderview) :
    QGraphicsScene(renderview),
    renderview_(renderview)
{
    update_timer_ = new QTimer;
    update_timer_->setSingleShot( true );

    connect( update_timer_.data(), SIGNAL(timeout()), SLOT(update()), Qt::QueuedConnection );
    connect( this, SIGNAL(sceneRectChanged ( const QRectF & )), SLOT(redraw()) );
    connect( renderview, SIGNAL(updatedCamera()), SLOT(updateMousePosInWorldCoordinates()), Qt::DirectConnection );
}

GraphicsScene::~GraphicsScene()
{
    delete update_timer_;

    QGraphicsScene::clear();
}

void GraphicsScene::
        drawBackground(QPainter *painter, const QRectF & rectf)
{
    if (!painter->device())
        return;

    double T = last_frame_.elapsedAndRestart();
    TIME_PAINTGL TaskTimer tt("GraphicsScene: Draw, last frame %.0f ms / %.0f fps", T*1e3, 1/T);
    if (update_timer_->isActive ())
        TaskInfo("GraphicsScene: Forced redraw");

    painter->beginNativePainting();

    glMatrixMode(GL_MODELVIEW);

    try { {
        glPushAttribContext attribs;
        glPushMatrixContext pmcp(GL_PROJECTION);
        glPushMatrixContext pmcm(GL_MODELVIEW);

        renderview_->initializeGL();

        float dpr = painter->device ()->devicePixelRatio();
        renderview_->model->render_settings.dpifactor = dpr;
        unsigned w = painter->device()->width();
        unsigned h = painter->device()->height();
        w *= dpr;
        h *= dpr;

        renderview_->setStates();

        {
            TIME_PAINTGL_DETAILS TaskTimer tt("glClear");
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        }

        QRect rect = QRectF(rectf.topLeft(), QSizeF(rectf.width ()-1, rectf.height ()-1)).toRect ();
        // QRect rect = tool_selector->parentTool()->geometry();
        rect.setWidth (rect.width ()*dpr);
        rect.setHeight (rect.height ()*dpr);
        rect.setLeft (rect.left ()*dpr);
        rect.setTop (rect.top ()*dpr);

        renderview_->resizeGL( rect, h );

        renderview_->paintGL();

        renderview_->defaultStates();

        }
        GlException_CHECK_ERROR();
    } catch (const std::exception& x) {
        TaskInfo("");
        TaskInfo(boost::format("std::exception\n%s") % boost::diagnostic_information(x));
        TaskInfo("");
    } catch (...) {
        TaskInfo(boost::format("Not an std::exception\n%s") % boost::current_exception_diagnostic_information ());
    }

    painter->endNativePainting();

    if (0 < draw_more_)
        draw_more_--;
    if (0 < draw_more_)
        update_timer_->start(5);
}

void GraphicsScene::
        drawForeground(QPainter *painter, const QRectF &)
{
}

bool GraphicsScene::
        event ( QEvent * e )
{
    DEBUG_EVENTS TaskTimer tt("GraphicsScene event %s %d", vartype(*e).c_str(), e->isAccepted());
    bool r = QGraphicsScene::event(e);
    DEBUG_EVENTS TaskTimer("GraphicsScene event %s info %d %d", vartype(*e).c_str(), r, e->isAccepted()).suppressTiming();
    return r;
}

bool GraphicsScene::
        eventFilter(QObject* o, QEvent* e)
{
    DEBUG_EVENTS TaskTimer tt("GraphicsScene eventFilter %s %s %d", vartype(*o).c_str(), vartype(*e).c_str(), e->isAccepted());
    bool r = QGraphicsScene::eventFilter(o, e);
    DEBUG_EVENTS TaskTimer("GraphicsScene eventFilter %s %s info %d %d", vartype(*o).c_str(), vartype(*e).c_str(), r, e->isAccepted()).suppressTiming();
    return r;
}

void GraphicsScene::
        mousePressEvent(QGraphicsSceneMouseEvent *e)
{
    DEBUG_EVENTS TaskTimer tt("GraphicsScene mousePressEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsScene::mousePressEvent(e);
    DEBUG_EVENTS TaskTimer("GraphicsScene mousePressEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void GraphicsScene::
        mouseMoveEvent(QGraphicsSceneMouseEvent *e)
{
    if (renderview_->model->render_settings.draw_cursor_marker)
        redraw();

    DEBUG_EVENTS TaskTimer tt("GraphicsScene mouseMoveEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsScene::mouseMoveEvent(e);
    DEBUG_EVENTS TaskTimer("GraphicsScene mouseMoveEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}

void GraphicsScene::
        mouseReleaseEvent(QGraphicsSceneMouseEvent *e)
{
    DEBUG_EVENTS TaskTimer tt("GraphicsScene mouseReleaseEvent %s %d", vartype(*e).c_str(), e->isAccepted());
    QGraphicsScene::mouseReleaseEvent(e);
    DEBUG_EVENTS TaskTimer("GraphicsScene mouseReleaseEvent %s info %d", vartype(*e).c_str(), e->isAccepted()).suppressTiming();
}


void GraphicsScene::
        redraw()
{
    if (0 == draw_more_)
    {
        draw_more_++;
        update_timer_->start(5);
    }
    else if (1 == draw_more_)
    {
        draw_more_++;
        // queue a redraw when finsihed drawing
    }
}


void GraphicsScene::
        updateMousePosInWorldCoordinates()
{
    Tools::Support::RenderViewInfo r(renderview_);
    Heightmap::Position cursorPos = r.getPlanePos( renderview_->glwidget->mapFromGlobal(QCursor::pos()) );
    renderview_->model->render_settings.cursor = GLvector(cursorPos.time, 0, cursorPos.scale);
}


} // namespace Tools
