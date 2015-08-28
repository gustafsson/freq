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
        Renderer(RenderBlock* render_block, BlockLayout block_size, glProjecter gl_projecter);
        ~Renderer();

        void renderBlock( pBlock ref, LevelOfDetail lod);

    private:
        RenderBlock* render_block;
        const glProjecter gl_projecter;
        unsigned prev_vbo;

        void draw(GLsizei n);
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

    GLint uniModelviewprojection=-2,
            uniModelview=-2,
            uniNormalMatrix=-2,
            attribVertex=-2;

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

    unsigned _mesh_width;
    unsigned _mesh_height;
    pVbo _mesh_position;

    struct ShaderData
    {
        ShaderData();
        ShaderData(const char* defines);

        ShaderPtr _shader_progp;
        unsigned _shader_prog=0;

        GLint uniModelviewprojection=-2,
                uniModelview=-2,
                uniNormalMatrix=-2;

        GLint uniVertText0=-2,
                uniVertText2=-2,
                uniColorTextureFactor=-2,
                uniFixedColor=-2,
                uniClearColor=-2,
                uniContourPlot=-2,
                uniFlatness=-2,
                uniYScale=-2,
                uniYOffset=-2,
                uniYNormalize=-2,
                uniLogScale=-2,
                uniScaleTex=-2,
                uniOffsTex=-2,
                uniTexDelta=-2,
                uniTexSize=-2,
                attribVertex=-2;

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
              u_tex_delta1=0,
              u_tex_delta2=0,
              u_texSize1=0,
              u_texSize2=0;

        void prepShader(BlockLayout block_size, RenderSettings* render_settings);
    };

    struct ShaderSettings
    {
        bool use_mipmap = false;
        bool draw3d = false;
        bool drawIsarithm = false;

        bool operator<(const ShaderSettings&) const;
    };

    ShaderData* getShader(ShaderSettings s);
    std::map<ShaderSettings,ShaderData> shaders;

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
