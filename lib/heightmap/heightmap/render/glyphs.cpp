#include "glyphs.h"

#include "glyphsfreetype.h"
#include "glyphsglut.h"
#include "glyphsfreetypeembedded.h"

namespace Heightmap {
namespace Render {

IGlyphs* GlyphFactory::
        makeIGlyphs()
{
#if defined(USE_GLUT) && defined(USE_FREETYPE_GL)
#error Use either glut or freetype-gl, not both.
#endif

#ifdef USE_GLUT
    return new GlyphsGlut;
#elif defined(USE_FREETYPE_GL)
    return new GlyphsFreetype;
#else
    // This is using an embedded font built by freetype-gl/makefont,
    // but it is self-contained and independent of the freetype library.
    // This is the default font engine.
    return new GlyphsFreetypeEmbedded;
#endif
}

} // namespace Render
} // namespace Heightmap
