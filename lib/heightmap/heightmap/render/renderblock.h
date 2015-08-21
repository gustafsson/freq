#ifndef HEIGHTMAP_RENDER_RENDERBLOCK_H
#define HEIGHTMAP_RENDER_RENDERBLOCK_H

#include "heightmap/block.h"
#include "vbo.h"
#include "rendersettings.h"
#include "GlTexture.h"
#include "glprojection.h"
#include "renderinfo.h"
#include "shaderresource.h"

typedef boost::shared_ptr<Vbo> pVbo;

namespace Heightmap {
namespace Render {

class RenderBlock
{
public:
    class Renderer : boost::noncopyable {
    public:
        Renderer(RenderBlock* render_block, BlockLayout block_size, glProjection gl_projection);
        ~Renderer();

        void renderBlock( pBlock ref, LevelOfDetail lod);

    private:
        RenderBlock* render_block;
        glProjection gl_projection;
        pVbo prev_vbo;

        void draw(unsigned tex_height, const pVbo& vbo);
    };

    RenderBlock(RenderSettings* render_settings);
    RenderBlock(const RenderBlock&)=delete;
    RenderBlock& operator=(const RenderBlock&)=delete;
    ~RenderBlock();

    void        init();
    bool        isInitialized();
    void        clearCaches();
    void        setSize( unsigned w, unsigned h);
    unsigned    trianglesPerBlock();

    GLint uniModelviewprojection=-1,
            uniModelview=-1,
            uniNormalMatrix=-1;

private:
    friend class RenderBlock::Renderer;

    enum InitializedLevel {
        NotInitialized,
        Initialized,
        InitializationFailed
    };

    InitializedLevel _initialized;

    RenderSettings* render_settings;
    RenderSettings::ColorMode _color_texture_colors;
    boost::shared_ptr<GlTexture> _colorTexture;

    ShaderPtr _shader_progp;
    unsigned _shader_prog;
    unsigned _mesh_width;
    unsigned _mesh_height;
    pVbo _mesh_position;

    GLint uniVertText0=-1,
            uniVertText2=-1,
            uniColorTextureFactor=-1,
            uniFixedColor=-1,
            uniClearColor=-1,
            uniContourPlot=-1,
            uniFlatness=-1,
            uniYScale=-1,
            uniYOffset=-1,
            uniYNormalize=-1,
            uniLogScale=-1,
            uniScaleTex=-1,
            uniOffsTex=-1,
            uniTexSize=-1;

    int   u_tex=0,
          u_tex_color=0;
    tvector<4, float>
          u_fixed_color,
          u_clearColor;
    float u_colorTextureFactor=0;
    bool  u_draw_contour_plot=false;
    float u_flatness=0,
          u_yScale=0,
          u_yOffset=0,
          u_yNormalize=0,
          u_logScale=0,
          u_logScale_x1=0,
          u_logScale_x2=0,
          u_scale_tex1=0,
          u_scale_tex2=0,
          u_offset_tex1=0,
          u_offset_tex2=0,
          u_texSize1=0,
          u_texSize2=0;

    // 1 << (subdivs-1) = max density of pixels per vertex
#ifdef GL_ES_VERSION_2_0
    static const int subdivs = 4;
#else
    static const int subdivs = 2;
#endif
    pVbo _mesh_index_buffer[subdivs*subdivs];

    void checkExtensions();
    void beginVboRendering(BlockLayout block_size);
    void endVboRendering();
    static void createMeshIndexBuffer(int w, int h, pVbo& vbo, int stepx=1, int stepy=1);
    void createMeshPositionVBO(int w, int h);
    void createColorTexture(unsigned N);
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERBLOCK_H
