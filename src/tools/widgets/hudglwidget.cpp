#include "hudglwidget.h"
#include "tools/renderview.h"
#include "gl.h"
#include "glPushContext.h"

namespace Tools {
namespace Widgets {

HudGlWidget::
        HudGlWidget (RenderView *view) :
    view_(view)
{
    connect(view_, SIGNAL(painting()), SLOT(painting()));
}


QRegion HudGlWidget::
        growRegion(const QRegion& r, int radius)
{
    QRegion m = r;

    for (int i=1; i<=radius; ++i)
        m |= m.translated(i,0)
            | m.translated(i,i)
            | m.translated(0,i)
            | m.translated(-i,i)
            | m.translated(-i,0)
            | m.translated(-i,-i)
            | m.translated(0,-i)
            | m.translated(i,-i);

    return m;
}


void HudGlWidget::
        painting ()
{
#ifdef LEGACY_OPENGL
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    // Set the viewport to the outer edges of this widget
    glViewport(
                viewport[0] + x(),
                viewport[1] + viewport[3] - (y() + height()),
                width(),
                height());

    glPushMatrixContext push_proj( GL_PROJECTION );
    glLoadIdentity();
    glOrtho( 0, 1, 0, 1, -1, 1);
    GlState::glDisable(GL_DEPTH_TEST);

    glPushMatrixContext push_model( GL_MODELVIEW );
    glLoadIdentity();

    this->paintWidgetGl2D();

    glViewport(viewport[0],viewport[1],viewport[2],viewport[3]);
#else
    EXCEPTION_ASSERTX(false, "requires LEGACY_OPENGL");
#endif
}


} // namespace Widgets
} // namespace Tools
