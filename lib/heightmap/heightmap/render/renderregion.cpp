#include "renderregion.h"

#include "unused.h"
#include "gl.h"
#include "glPushContext.h"

namespace Heightmap {
namespace Render {

RenderRegion::RenderRegion(Region r)
    :
      r(r)
{
}


void RenderRegion::
        render()
{
    // if (!renderBlock(...) && (0 == "render red warning cross" || render_settings->y_scale < yscalelimit))
    //float y = _frustum_clip.projectionPlane[1]*.05;
    float y = 0.05f;

    UNUSED(glPushMatrixContext mc)( GL_MODELVIEW );

    glTranslatef(r.a.time, 0, r.a.scale);
    glScalef(r.time(), 1, r.scale());

    UNUSED(glPushAttribContext attribs);

    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);
    glBindTexture(GL_TEXTURE_2D, 0);
    glColor4f( 0.8f, 0.2f, 0.2f, 0.5f );
    glLineWidth(2);

    glBegin(GL_LINE_STRIP);
        glVertex3f( 0, 0, 0 );
        glVertex3f( 1, 0, 1 );
        glVertex3f( 1, 0, 0 );
        glVertex3f( 0, 0, 1 );
        glVertex3f( 0, 0, 0 );
        glVertex3f( 1, 0, 0 );
        glVertex3f( 1, 0, 1 );
        glVertex3f( 0, 0, 1 );
    glEnd();
    glColor4f( 0.2f, 0.8f, 0.8f, 0.5f );
    glBegin(GL_LINE_STRIP);
        glVertex3f( 0, y, 0 );
        glVertex3f( 1, y, 1 );
        glVertex3f( 1, y, 0 );
        glVertex3f( 0, y, 1 );
        glVertex3f( 0, y, 0 );
        glVertex3f( 1, y, 0 );
        glVertex3f( 1, y, 1 );
        glVertex3f( 0, y, 1 );
    glEnd();
}


} // namespace Render
} // namespace Heightmap
