#ifndef HEIGHTMAP_RENDER_GLYPHSFREETYPEEMBEDDED_H
#define HEIGHTMAP_RENDER_GLYPHSFREETYPEEMBEDDED_H

#include "glyphs.h"
#include "tvector.h"

namespace Heightmap {
namespace Render {

class GlyphsFreetypeEmbedded : public Heightmap::Render::IGlyphs
{
public:
    GlyphsFreetypeEmbedded();

    void drawGlyphs( const glProjection& projection, const std::vector<GlyphData>& data) override;

private:
    float print( const wchar_t *text, float letter_spacing=0.f );
    void buildGlyphs(const std::vector<GlyphData>& data);

    struct Glyph {
        float s,t;
        tvector<4,GLfloat> p;
    };

    std::vector<Glyph> glyphs;
    std::unique_ptr<QOpenGLShaderProgram> program_;

    GLuint texid;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_GLYPHSFREETYPEEMBEDDED_H
