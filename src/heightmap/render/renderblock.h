#ifndef HEIGHTMAP_RENDER_RENDERBLOCK_H
#define HEIGHTMAP_RENDER_RENDERBLOCK_H

#include "heightmap/block.h"
#include "vbo.h"
#include "../rendersettings.h"
#include "GlTexture.h"

typedef boost::shared_ptr<Vbo> pVbo;

namespace Heightmap {
namespace Render {

class RenderBlock
{
public:
    class Renderer : boost::noncopyable {
    public:
        Renderer(RenderBlock* render_block, BlockLayout block_size);
        ~Renderer();

        void renderBlock( pBlock ref );

    private:
        unsigned vbo_size;
        RenderSettings render_settings;
    };

    RenderBlock(RenderSettings* render_settings);

    void        init();
    void        clearCaches();
    void        setSize( unsigned w, unsigned h);
    unsigned    trianglesPerBlock();

private:
    friend class RenderBlock::Renderer;

    RenderSettings* render_settings;
    RenderSettings::ColorMode _color_texture_colors;
    boost::shared_ptr<GlTexture> _colorTexture;

    unsigned _shader_prog;
    unsigned _mesh_index_buffer;
    unsigned _mesh_width;
    unsigned _mesh_height;
    unsigned _vbo_size;
    pVbo _mesh_position;

    void beginVboRendering(BlockLayout block_size);
    static void endVboRendering();
    void createMeshIndexBuffer(int w, int h);
    void createMeshPositionVBO(int w, int h);
    void createColorTexture(unsigned N);
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERBLOCK_H
