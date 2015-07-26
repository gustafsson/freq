#include "renderfrustum.h"
#include "frustumclip.h"

// gpumisc
#include "gl.h"
#include "glPushContext.h"

namespace Heightmap {
namespace Render {

RenderFrustum::
        RenderFrustum(const glProjection& gl_projection)
{
    Render::FrustumClip frustum(gl_projection);
    clippedFrustum = frustum.visibleXZ ();
    camera = frustum.getCamera ();
}


void RenderFrustum::
        drawFrustum()
{
#ifdef LEGACY_OPENGL
    if (clippedFrustum.empty())
        return;

    vectord closest = clippedFrustum.front();
    for ( std::vector<vectord>::const_iterator i = clippedFrustum.begin();
            i!=clippedFrustum.end();
            i++)
    {
        if ((closest - camera).dot() > (*i - camera).dot())
            closest = *i;
    }


    glPushAttribContext ac;

    glDisable(GL_DEPTH_TEST);

    glPushMatrixContext mc(GL_MODELVIEW);

    glEnable(GL_BLEND);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_FLOAT, 0, &clippedFrustum[0]);


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
#endif // LEGACY_OPENGL
}

} // namespace Render
} // namespace Heightmap
