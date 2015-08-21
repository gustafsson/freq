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
    ~GlyphsFreetypeEmbedded();

    void drawGlyphs( const glProjection& projection, const std::vector<GlyphData>& data) override;

private:
    float print( const wchar_t *text, float letter_spacing=0.f );
    void buildGlyphs(const std::vector<GlyphData>& data);

    struct Glyph {
        tvector<4,GLfloat> p;
        float s,t;
    };

    std::vector<Glyph> glyphs;
    std::vector<tvector<4,GLfloat>> quad_v;
    std::unique_ptr<QOpenGLShaderProgram> program_;
    std::unique_ptr<QOpenGLShaderProgram> overlay_program_;

    GLuint texid;
    GLuint glyphbuffer_=0;
    size_t glyphbuffer_size = 0;
    GLuint vertexbuffer_=0;
    size_t vertexbuffer_size = 0;

    GLuint program_qt_ProjectionMatrixLocation_ = -1,
           program_qt_ModelViewVertexLocation_ = -1,
           program_qt_MultiTexCoord0Location_ = -1,
           overlay_program_qt_ProjectionMatrixLocation_ = -1;
};

} // namespace Render
} // namespace Heightmap

#endif // HEIGHTMAP_RENDER_GLYPHSFREETYPEEMBEDDED_H
