/**
  Based on embedded-font.c in freetype-gl demos.
  */

#define VERA32

#include "glyphsfreetypeembedded.h"
#ifdef VERA32
#include "vera_32.h"
#else
#include "vera_16.h"
#endif
#include "tvector.h"
#include "log.h"
#include "gluperspective.h"
#include "GlException.h"
#include "shaderresource.h"
#include "exceptionassert.h"
#include "glstate.h"

#include <QOpenGLShaderProgram>

//#define vera32 font

namespace Heightmap {
namespace Render {

float GlyphsFreetypeEmbedded::
        print( const wchar_t *text, float letter_spacing )
{
#ifdef VERA32
    texture_font_t& vera = vera_32;
#else
    texture_font_t& vera = vera_16;
#endif

    float pen_x = 0, pen_y = 0;
    auto add = [this](float s, float t, float x, float y) {
        tvector<4,GLfloat> v{x,y,0,1};
        glyphs.push_back (Glyph{v, s, t});
    };

    size_t i, j;
    size_t N = wcslen(text);
    glyphs.reserve (glyphs.size ()+N*6);

    for( i=0; i<N; ++i)
    {
        texture_glyph_t *glyph = 0;
        for( j=0; j<vera.glyphs_count; ++j)
        {
            if( vera.glyphs[j].charcode == text[i] )
            {
                glyph = &vera.glyphs[j];
                break;
            }
        }
        if( !glyph )
        {
            continue;
        }

        float x = pen_x + glyph->offset_x;
        float y = pen_y + glyph->offset_y;
        float w  = glyph->width;
        float h  = glyph->height;
        add( glyph->s0, glyph->t0, x,   y );
        add( glyph->s0, glyph->t1, x,   y-h );
        add( glyph->s1, glyph->t1, x+w, y-h );
        add( glyph->s0, glyph->t0, x,   y );
        add( glyph->s1, glyph->t1, x+w, y-h );
        add( glyph->s1, glyph->t0, x+w, y );
        pen_x += glyph->advance_x * ( 1.f + letter_spacing );
        pen_y += glyph->advance_y;
    }

    return pen_x;
}


GlyphsFreetypeEmbedded::
        GlyphsFreetypeEmbedded()
    :
      texid(0)
{
    QOpenGLFunctions::initializeOpenGLFunctions ();

    glGenTextures( 1, &texid );
    glBindTexture( GL_TEXTURE_2D, texid );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR );
#ifdef VERA32
    texture_font_t& vera = vera_32;
#else
    texture_font_t& vera = vera_16;
#endif
    glTexImage2D( GL_TEXTURE_2D, 0, GL_RED, vera.tex_width, vera.tex_height,
                  0, GL_RED, GL_UNSIGNED_BYTE, vera.tex_data );
    glGenerateMipmap (GL_TEXTURE_2D);
}


GlyphsFreetypeEmbedded::
        ~GlyphsFreetypeEmbedded()
{
    if (!QOpenGLContext::currentContext ()) {
        Log ("%s: destruction without gl context leaks vbos %d and %d") % __FILE__ % glyphbuffer_ % vertexbuffer_;
        return;
    }

    if (glyphbuffer_)
        GlState::glDeleteBuffers (1, &glyphbuffer_);
    if (vertexbuffer_)
        GlState::glDeleteBuffers (1, &vertexbuffer_);
}


void GlyphsFreetypeEmbedded::
        buildGlyphs(const std::vector<GlyphData>& glyphdata)
{
    glyphs.clear ();
    size_t gi = 0;

    typedef tvector<4,GLfloat> v4f;
    quad_v.resize( 6*glyphdata.size () );
    v4f* quad = &quad_v[0];

#ifdef VERA32
    // texture detail, how big a character is in the text_buffer_add_text coordinate system
    float f = 32/1.5;
#else
    float f = 16/1.5;
#endif

    for (const GlyphData& g : glyphdata) {
        std::wstring text;
        std::copy(g.text.begin (), g.text.end (), std::inserter(text,text.begin()));

        double w = print(text.c_str (), g.letter_spacing);

        matrixd modelview { g.modelview };
        modelview *= matrixd::scale (1/f,1/f,1.);
        modelview *= matrixd::translate (g.margin*f - w*g.align_x, -g.align_y*f, 0);

        for (;gi<glyphs.size ();gi++)
            glyphs[gi].p = modelview * glyphs[gi].p;

        float z = .3*f;
        float q = .3*f;
        // these matrix multiplications would potentially be faster on the GPU,
        // but as the different glyphs have different matrices it's faster to
        // to do the matrix-vector multiplication on the CPU
        *quad++ = modelview * v4f(0 - z, 0 - q, 0, 1);
        *quad++ = modelview * v4f(w + z, 0 - q, 0, 1);
        *quad++ = modelview * v4f(w + z, f + q, 0, 1);
        *quad++ = modelview * v4f(0 - z, 0 - q, 0, 1);
        *quad++ = modelview * v4f(w + z, f + q, 0, 1);
        *quad++ = modelview * v4f(0 - z, f + q, 0, 1);
    }
}


void GlyphsFreetypeEmbedded::
        drawGlyphs( const glProjection& gl_projection, const std::vector<GlyphData>& data )
{
    if (!vertexbuffer_)
    {
        glGenBuffers(1, &vertexbuffer_);
        glGenBuffers(1, &glyphbuffer_);
        program_ = ShaderResource::loadGLSLProgramSource (
                                          R"vertexshader(
                                              attribute highp vec4 qt_ModelViewVertex;
                                              attribute highp vec2 qt_MultiTexCoord0;
                                              uniform highp mat4 qt_ProjectionMatrix;
                                              varying highp vec2 texcoord;

                                              void main() {
                                                  gl_Position = qt_ProjectionMatrix * qt_ModelViewVertex;
                                                  texcoord = qt_MultiTexCoord0;
                                              }

                                          )vertexshader",
                                          R"fragmentshader(
                                              uniform highp sampler2D tex;
                                              uniform highp vec4 qt_Color;
                                              varying highp vec2 texcoord;

                                              void main() {
                                                  highp vec4 c = qt_Color;
                                                  c.a *= texture2D(tex, texcoord).r;
                                                  gl_FragColor = c;
                                              }
                                           )fragmentshader");
        GlState::glUseProgram (program_->programId());
        program_->setUniformValue("tex", 0);
        program_->setUniformValue("qt_Color", 0, 0, 0, 0.8);
        program_qt_ProjectionMatrixLocation_ = program_->uniformLocation("qt_ProjectionMatrix");
        program_qt_ModelViewVertexLocation_ = program_->attributeLocation("qt_ModelViewVertex");
        program_qt_MultiTexCoord0Location_ = program_->attributeLocation("qt_MultiTexCoord0");
        GlState::glUseProgram (0);

        overlay_program_ = ShaderResource::loadGLSLProgramSource (
                                          R"vertexshader(
                                              attribute highp vec4 qt_ModelViewVertex;
                                              uniform highp mat4 qt_ProjectionMatrix;
                                              varying highp vec2 texcoord;

                                              void main() {
                                                  gl_Position = qt_ProjectionMatrix * qt_ModelViewVertex;
                                              }

                                          )vertexshader",
                                          R"fragmentshader(
                                              uniform highp vec4 qt_Color;

                                              void main() {
                                                  highp vec4 c = qt_Color;
                                                  gl_FragColor = c;
                                              }
                                           )fragmentshader");

        GlState::glUseProgram (overlay_program_->programId());
        overlay_program_->setUniformValue("qt_Color", 1, 1, 1, 0.5);
        overlay_program_qt_ProjectionMatrixLocation_ = overlay_program_->uniformLocation("qt_ProjectionMatrix");
        GlState::glUseProgram (0);
    }

    if (!program_ || !program_->isLinked ())
        return;

    buildGlyphs(data);

    GlState::glEnable( GL_BLEND );
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );

    {
        GlState::glBindBuffer(GL_ARRAY_BUFFER, vertexbuffer_);
        if (quad_v.size () > vertexbuffer_size || vertexbuffer_size > quad_v.size ()*4)
        {
            glBufferData(GL_ARRAY_BUFFER, sizeof(tvector<4,GLfloat>)*quad_v.size (), &quad_v[0], GL_STREAM_DRAW);
            vertexbuffer_size = quad_v.size ();
        }
        else
        {
            glBufferSubData (GL_ARRAY_BUFFER, 0, sizeof(tvector<4,GLfloat>)*quad_v.size (), &quad_v[0]);
        }

        GlState::glUseProgram (overlay_program_->programId());
        overlay_program_->setUniformValue(overlay_program_qt_ProjectionMatrixLocation_,
                                  QMatrix4x4(GLmatrixf(gl_projection.projection).transpose ().v ()));

        GlState::glEnableVertexAttribArray (0);
        glVertexAttribPointer( 0, 4, GL_FLOAT, GL_FALSE, 0, 0 );
        GlState::glDrawArrays (GL_TRIANGLES, 0, quad_v.size());
    }

    {
        GlState::glBindBuffer(GL_ARRAY_BUFFER, glyphbuffer_);
        if (glyphs.size () > glyphbuffer_size || glyphbuffer_size > glyphs.size ()*4)
        {
            glBufferData(GL_ARRAY_BUFFER, sizeof(Glyph)*glyphs.size (), &glyphs[0], GL_STREAM_DRAW);
            glyphbuffer_size = glyphs.size ();
        }
        else
        {
            glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(Glyph)*glyphs.size (), &glyphs[0]);
        }

        GlState::glUseProgram (program_->programId());

        program_->setUniformValue(program_qt_ProjectionMatrixLocation_,
                                  QMatrix4x4(GLmatrixf(gl_projection.projection).transpose ().v ()));

        GlState::glEnableVertexAttribArray (1);
        program_->setAttributeBuffer(program_qt_ModelViewVertexLocation_, GL_FLOAT, 0, 4, sizeof(Glyph));
        program_->setAttributeBuffer(program_qt_MultiTexCoord0Location_, GL_FLOAT, sizeof(tvector<4,GLfloat>), 2, sizeof(Glyph));

        glBindTexture (GL_TEXTURE_2D, texid);
        GlState::glDrawArrays (GL_TRIANGLES, 0, glyphs.size());

        GlState::glDisableVertexAttribArray (1);
        GlState::glDisableVertexAttribArray (0);
        GlState::glBindBuffer (GL_ARRAY_BUFFER, 0);
    }

    GlException_SAFE_CALL( GlState::glUseProgram (0) );

    GlState::glDisable (GL_BLEND);
}


} // namespace Render
} // namespace Heightmap

