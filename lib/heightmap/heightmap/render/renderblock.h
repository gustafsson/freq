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

    GLuint uniModelviewprojection=0,
            uniModelview=0,
            uniNormalMatrix=0;

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

    GLint uniVertText0=0,
            uniVertText2=0,
            uniColorTextureFactor=0,
            uniFixedColor=0,
            uniClearColor=0,
            uniContourPlot=0,
            uniFlatness=0,
            uniYScale=0,
            uniYOffset=0,
            uniYNormalize=0,
            uniLogScale=0,
            uniScaleTex=0,
            uniOffsTex=0,
            uniTexSize=0;

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
