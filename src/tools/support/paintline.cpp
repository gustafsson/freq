#include "paintline.h"
#include "toolglbrush.h"
#include "GlException.h"

namespace Tools {
namespace Support {

void PaintLine::
        drawSlice(unsigned N, Heightmap::Position* pts, float r, float g, float b, float a)
{
    if (0==N)
        return;
#ifdef LEGACY_OPENGL
    GlException_CHECK_ERROR();

    ToolGlBrush tgb;
    glColor4f( r, g, b, a);
    float y = 1;

    glBegin(GL_TRIANGLE_STRIP);
    for (unsigned k=0; k<N; k++)
    {
        glVertex3f( pts[k].time, 0, pts[k].scale );
        glVertex3f( pts[k].time, y, pts[k].scale );
    }
    glEnd();

    glLineWidth(1.6f);
    glBegin(GL_LINE_STRIP);
    for (unsigned k=0; k<N; k++)
    {
        glVertex3f( pts[k].time, y, pts[k].scale );
    }
    glEnd();
    glLineWidth(0.5f);

    GlException_CHECK_ERROR();
#endif // LEGACY_OPENGL
}

} // namespace Support
} // namespace Tools
