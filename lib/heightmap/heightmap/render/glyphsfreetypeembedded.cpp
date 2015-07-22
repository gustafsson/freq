/**
  Based on embedded-font.c in freetype-gl demos.
  */

#include "glyphsfreetypeembedded.h"
#include "vera32.h"
#include "tvector.h"
#include "log.h"
#include "gluperspective.h"

#include <QOpenGLShaderProgram>

namespace Heightmap {
namespace Render {

float GlyphsFreetypeEmbedded::
        print_at( const wchar_t *text, float letter_spacing )
{
    float pen_x = 0, pen_y = 0;
    auto add = [this](float s, float t, float x, float y) {
        tvector<4,GLfloat> v{x,y,0,1};
        glyphs.push_back (Glyph{s, t, tvector<4,GLfloat>(v)});
    };

    size_t i, j;
    size_t N = wcslen(text);
    for( i=0; i<N; ++i)
    {
        texture_glyph_t *glyph = 0;
        for( j=0; j<vera32.glyphs_count; ++j)
        {
            if( vera32.glyphs[j].charcode == text[i] )
            {
                glyph = &vera32.glyphs[j];
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
        add( glyph->s0, glyph->t0, x, y );
        add( glyph->s0, glyph->t1, x,   y-h );
        add( glyph->s1, glyph->t1, x+w, y-h );
        add( glyph->s0, glyph->t0, x,   y );
        add( glyph->s1, glyph->t1, x+w, y-h );
        add( glyph->s1, glyph->t0, x+w, y );
        pen_x += glyph->advance_x + letter_spacing;
        pen_y += glyph->advance_y;

    }

    return pen_x;
}


GlyphsFreetypeEmbedded::
        GlyphsFreetypeEmbedded()
    :
      texid(0)
{
    glGenTextures( 1, &texid );
    glBindTexture( GL_TEXTURE_2D, texid );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexImage2D( GL_TEXTURE_2D, 0, GL_ALPHA, vera32.tex_width, vera32.tex_height,
                  0, GL_ALPHA, GL_UNSIGNED_BYTE, vera32.tex_data );
    glBindTexture( GL_TEXTURE_2D, 0 );
}


void GlyphsFreetypeEmbedded::
        buildGlyphs(const std::vector<GlyphData>& glyphdata)
{
    glyphs.clear ();

    typedef tvector<2,GLfloat> GLvector2F;
    std::vector<GLvector2F> quad(4);

    size_t gi = 0;
    float s = 0.045, is = 1/s;
    for (const GlyphData& g : glyphdata) {
        double w = g.margin*is;
        double letter_spacing = g.letter_spacing*is;
        std::wstring text;
        std::copy(g.text.begin (), g.text.end (), std::inserter(text,text.begin()));
        w += print_at(text.c_str (), letter_spacing);

        matrixd modelview = g.modelview;
        modelview *= matrixd::scale (s,s,1.);
        modelview *= matrixd::translate (-w*g.align_x,-g.align_y*is,0);

        for (;gi<glyphs.size ();gi++)
            glyphs[gi].p = tvector<4,GLfloat>(modelview * glyphs[gi].p);

        glLoadMatrixd (modelview.v ());
        float z = .1*is;
        float q = .2*is;
        glEnableClientState(GL_VERTEX_ARRAY);
        quad[0] = GLvector2F(0 - z, 0 - q);
        quad[1] = GLvector2F(w + z, 0 - q);
        quad[2] = GLvector2F(0 - z, is + q);
        quad[3] = GLvector2F(w + z, is + q);
        glVertexPointer(2, GL_FLOAT, 0, &quad[0]);
        glColor4f(1,1,1,0.5);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, quad.size());
        glDisableClientState(GL_VERTEX_ARRAY);
    }
}


void GlyphsFreetypeEmbedded::
        drawGlyphs( const glProjection& gl_projection, const std::vector<GlyphData>& data )
{
    if (!program_)
    {
        program_.reset (new QOpenGLShaderProgram());
        program_->addShaderFromSourceCode(QOpenGLShader::Vertex,
                                          R"vertexshader(
                                              attribute highp vec2 qt_MultiTexCoord0;
                                              attribute highp vec4 qt_Vertex;
                                              uniform highp mat4 qt_ModelViewMatrix;
                                              uniform highp mat4 qt_ProjectionMatrix;
                                              varying highp vec2 texcoord;

                                              void main() {
                                                  gl_Position = qt_ProjectionMatrix * qt_ModelViewMatrix * qt_Vertex;
                                                  texcoord = qt_MultiTexCoord0;
                                              }

                                          )vertexshader");
        program_->addShaderFromSourceCode(QOpenGLShader::Fragment,
                                          R"fragmentshader(
                                              uniform highp sampler2D tex;
                                              uniform highp vec4 qt_Color;
                                              varying highp vec2 texcoord;

                                              void main() {
                                                  vec4 c = qt_Color;
                                                  c.a *= texture2D(tex, texcoord).a;
                                                  gl_FragColor = c;
                                              }
                                           )fragmentshader");

        program_->bindAttributeLocation("qt_MultiTexCoord0", 0);
        program_->bindAttributeLocation("qt_Vertex", 1);

        if (!program_->link())
            Log("glyphsfreetypeembedded: invalid shader\n%s")
                    % program_->log ().toStdString ();
    }

    if (!program_->isLinked ())
        return;

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd (gl_projection.projection.v ());
    glMatrixMode(GL_MODELVIEW);

    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    buildGlyphs(data);

    program_->bind();

    program_->setUniformValue("tex", 0);
    program_->setUniformValue("qt_Color", 0, 0, 0, 0.8);
    program_->setUniformValue("qt_ProjectionMatrix",
                              QMatrix4x4(GLmatrixf(gl_projection.projection).transpose ().v ()));
    program_->setUniformValue("qt_ModelViewMatrix",
                              QMatrix4x4(GLmatrixf::identity ().v ()));

    program_->enableAttributeArray(0);
    program_->enableAttributeArray(1);

    program_->setAttributeArray(0, GL_FLOAT, &glyphs[0].s, 2, sizeof(Glyph));
    program_->setAttributeArray(1, GL_FLOAT, &glyphs[0].p[0], 4, sizeof(Glyph));

    glBindTexture( GL_TEXTURE_2D, texid );
    glDrawArrays(GL_TRIANGLES, 0, glyphs.size());
    glBindTexture( GL_TEXTURE_2D, 0 );

    program_->disableAttributeArray (0);
    program_->disableAttributeArray (1);
    program_->release();
}


} // namespace Render
} // namespace Heightmap

