#include "glyphs.h"

#include "glyphsfreetype.h"
#include "glyphsglut.h"

namespace Heightmap {
namespace Render {

IGlyphs* GlyphFactory::
        makeIGlyphs()
{
    return new GlyphsGlut;
//    return new GlyphsFreetype;
}

} // namespace Render
} // namespace Heightmap

