#ifndef HEIGHTMAP_RENDER_RENDERBLOCK_H
#define HEIGHTMAP_RENDER_RENDERBLOCK_H

#include "heightmap/block.h"
#include "vbo.h"
#include "rendersettings.h"
#include "GlTexture.h"
#include "glprojection.h"

typedef boost::shared_ptr<Vbo> pVbo;

namespace Heightmap {
namespace Render {

class RenderBlock
{
public:
    typedef std::shared_ptr<RenderBlock> ptr;

    class Renderer : boost::noncopyable {
    public:
        Renderer(RenderBlock* render_block, BlockLayout block_size, glProjection gl_projection);
        ~Renderer();

        void renderBlock( pBlock ref );

    private:
        unsigned vbo_size;
        RenderSettings render_settings;
        glProjection gl_projection;
        unsigned uniModelviewprojection = 0;
        unsigned uniModelview = 0;
        unsigned uniNormalMatrix = 0;

        void draw(unsigned tex_height);
    };

    RenderBlock(RenderSettings* render_settings);

    void        init();
    bool        isInitialized();
    void        clearCaches();
    void        setSize( unsigned w, unsigned h);
    unsigned    trianglesPerBlock();

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

    unsigned _shader_prog;
    unsigned _mesh_index_buffer;
    unsigned _mesh_width;
    unsigned _mesh_height;
    unsigned _vbo_size;
    pVbo _mesh_position;

    void checkExtensions();
    void beginVboRendering(BlockLayout block_size);
    static void endVboRendering();
    void createMeshIndexBuffer(int w, int h);
    void createMeshPositionVBO(int w, int h);
    void createColorTexture(unsigned N);
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERBLOCK_H
