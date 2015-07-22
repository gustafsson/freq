#include "glyphs.h"

#include "glyphsfreetype.h"
#include "glyphsglut.h"
#include "glyphsfreetypeembedded.h"

namespace Heightmap {
namespace Render {

IGlyphs* GlyphFactory::
        makeIGlyphs()
{
    //    return new GlyphsGlut;
    //    return new GlyphsFreetype;
    return new GlyphsFreetypeEmbedded;
}

} // namespace Render
} // namespace Heightmap

