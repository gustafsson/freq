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
    RenderBlock(RenderSettings* render_settings);

    void init();
    void clearCaches();
    bool renderBlock( pBlock ref, bool draw_flat );
    void renderBlockError( BlockLayout block_size, Region r, float y );

// private:
    void beginVboRendering(BlockLayout block_size, unsigned frame_number);
    void endVboRendering();

    void setSize( unsigned w, unsigned h);
    unsigned trianglesPerBlock();

private:
    RenderSettings* render_settings;
    RenderSettings::ColorMode _color_texture_colors;
    boost::shared_ptr<GlTexture> _colorTexture;

    bool _drawcrosseswhen0;
    unsigned _shader_prog;
    unsigned _mesh_index_buffer;
    unsigned _mesh_width;
    unsigned _mesh_height;
    unsigned _vbo_size;
    pVbo _mesh_position;
    unsigned _frame_number;

    void createMeshIndexBuffer(int w, int h);
    void createMeshPositionVBO(int w, int h);
    void createColorTexture(unsigned N);
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_RENDERBLOCK_H
