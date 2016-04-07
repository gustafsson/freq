#ifndef HEIGHTMAP_RENDER_GLYPHSFREETYPE_H
#define HEIGHTMAP_RENDER_GLYPHSFREETYPE_H

#include "glyphs.h"
#include <memory>

namespace Heightmap {
namespace Render {

class GlyphsFreetype: public IGlyphs
{
public:
    GlyphsFreetype();
    ~GlyphsFreetype();

    void drawGlyphs( const glProjection& projection, const std::vector<GlyphData>& ) override;

private:
    void buildGlyphs(const std::vector<GlyphData>& data);

    std::unique_ptr<struct GlyphsFreetypePrivate> p;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_GLYPHSFREETYPE_H
