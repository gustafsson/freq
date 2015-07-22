#ifndef HEIGHTMAP_RENDER_GLYPHSFREETYPE_H
#define HEIGHTMAP_RENDER_GLYPHSFREETYPE_H

#include "glyphs.h"

namespace Heightmap {
namespace Render {

class GlyphsFreetype: public IGlyphs
{
public:
    GlyphsFreetype();

    void drawGlyphs( const matrixd& projection, const std::vector<GlyphData>& ) override;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_GLYPHSFREETYPE_H
