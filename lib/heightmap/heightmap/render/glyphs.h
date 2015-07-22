#ifndef HEIGHTMAP_RENDER_GLYPHS_H
#define HEIGHTMAP_RENDER_GLYPHS_H

#include "GLvector.h"

namespace Heightmap {
namespace Render {

struct GlyphData {
    matrixd modelview;
    std::string text;
    double margin;
    double letter_spacing;
    double align_x;
    double align_y;
};


class IGlyphs
{
public:
    virtual ~IGlyphs() {}

    virtual void drawGlyphs( const matrixd& projection, const std::vector<GlyphData>& ) = 0;
};


class GlyphFactory
{
public:
    static IGlyphs* makeIGlyphs();
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_GLYPHS_H
