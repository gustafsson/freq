#include "renderfrustum.h"

// gpumisc
#include "gl.h"
#include "glPushContext.h"

namespace Heightmap {
namespace Render {

RenderFrustum::
        RenderFrustum(RenderSettings& render_settings, std::vector<GLvector> clippedFrustum)
    :
      render_settings(render_settings),
      clippedFrustum(clippedFrustum)
{
}


void RenderFrustum::
        drawFrustum()
{
    if (clippedFrustum.empty())
        return;

    GLvector closest = clippedFrustum.front();
    for ( std::vector<GLvector>::const_iterator i = clippedFrustum.begin();
            i!=clippedFrustum.end();
            i++)
    {
        if ((closest - render_settings.camera).dot() > (*i - render_settings.camera).dot())
            closest = *i;
    }


    glPushAttribContext ac;

    glDisable(GL_DEPTH_TEST);

    glPushMatrixContext mc(GL_MODELVIEW);

    glEnable(GL_BLEND);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_DOUBLE, 0, &clippedFrustum[0]);


    // dark inside
    float darkness = 0.2f; // 0 = not dark, 1 = very dark
    glColor4f( darkness, darkness, darkness, 1 );
    glBlendEquation( GL_FUNC_REVERSE_SUBTRACT );
    glBlendFunc( GL_ONE_MINUS_DST_COLOR, GL_ONE );
    glDrawArrays( GL_TRIANGLE_FAN, 0, clippedFrustum.size() );
    glBlendEquation( GL_FUNC_ADD );


    // black border
    glColor4f( 0, 0, 0, 0.5 );
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth( 0.5 );
    glDrawArrays(GL_LINE_LOOP, 0, clippedFrustum.size());


    glDisableClientState(GL_VERTEX_ARRAY);
    glDisable(GL_BLEND);
}

} // namespace Render
} // namespace Heightmap
