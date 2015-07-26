#ifndef HEIGHTMAP_RENDER_GLYPHSGLUT_H
#define HEIGHTMAP_RENDER_GLYPHSGLUT_H

#include "glyphs.h"

namespace Heightmap {
namespace Render {

class GlyphsGlut: public IGlyphs
{
#if defined(USE_GLUT) && defined(LEGACY_OPENGL)
public:
    GlyphsGlut();

    void drawGlyphs( const glProjection& gl_projection, const std::vector<GlyphData>& ) override;
#endif
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_GLYPHSGLUT_H
