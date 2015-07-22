#ifndef HEIGHTMAP_RENDER_GLYPHSGLUT_H
#define HEIGHTMAP_RENDER_GLYPHSGLUT_H

#include "glyphs.h"

namespace Heightmap {
namespace Render {

class GlyphsGlut: public IGlyphs
{
public:
    GlyphsGlut();

    void drawGlyphs( const glProjection& gl_projection, const std::vector<GlyphData>& ) override;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_GLYPHSGLUT_H
