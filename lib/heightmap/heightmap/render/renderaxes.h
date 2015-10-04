#ifndef HEIGHTMAP_RENDER_RENDERAXES_H
#define HEIGHTMAP_RENDER_RENDERAXES_H

#include "rendersettings.h"
#include "frustumclip.h"
#include "heightmap/freqaxis.h"
#include "tvector.h"
#include "glyphs.h"

#include "glprojection.h"

#include <vector>

namespace Heightmap {
namespace Render {

/**
 * @brief The RenderAxes class should compute which glyphs to draw along the frustum edges, and then ask IGlyphs to draw them.
 */
class RenderAxes
{
public:
    struct Vertex {
        tvector<4,GLfloat> position;
        tvector<4,GLfloat> color;
    };

    struct AxesElements {
        std::vector<GlyphData> glyphs;
        std::vector<Vertex> vertices;
        std::vector<Vertex> orthovertices;
    };

    RenderAxes();
    ~RenderAxes();

    void drawAxes( const RenderSettings* render_settings,
                   const glProjection* gl_projection,
                   FreqAxis display_scale, float T );

private:
    struct Side {
        vectord const p1;
        vectord const p2;
        vectord::T const timePerPixel1;
        vectord::T const scalePerPixel1;
        vectord::T const timePerPixel2;
        vectord::T const scalePerPixel2;
    };

    void getElements( RenderAxes::AxesElements& ae, float T );
    template<bool taxis>
    void drawSide( RenderAxes::AxesElements& ae, const Side& s, const vectord::T timePerPixel_closest, const vectord::T scalePerPixel_closest, const vectord v0, const vectord inside );
    void drawPiano( RenderAxes::AxesElements& ae, const Side& s, const vectord& p1_0, const vectord& inside );

    void drawElements( const AxesElements& );

    const RenderSettings* render_settings;
    const glProjection* gl_projection;
    FreqAxis display_scale;
    AxesElements ae_;
    std::unique_ptr<QOpenGLShaderProgram> program_, orthoprogram_;
    IGlyphs* glyphs_;
    GLuint orthobuffer_=0, vertexbuffer_=0;
    size_t orthobuffer_size_=0, vertexbuffer_size_=0;

    int uni_ProjectionMatrix=-1, uni_ModelViewMatrix=-1,
        attrib_Vertex=-1, attrib_Color=-1,
        uni_OrthoProjectionMatrix=-1,
        attrib_OrthoVertex=-1, attrib_OrthoColor=-1;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERAXES_H
