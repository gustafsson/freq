/**
  Based on subpixel.c demo in freetype-gl
  */

#ifdef USE_FREETYPE_GL

#include "glyphsfreetype.h"
#include "log.h"

#include <ft2build.h>
#include FT_CONFIG_OPTIONS_H

#include <stdio.h>
#include <wchar.h>

#include "freetype-gl.h"

#include "vertex-buffer.h"
#include "text-buffer.h"
#include "markup.h"
#include "shader.h"
#include "mat4.h"
#include "shaderresource.h"

namespace Heightmap {
namespace Render {

// ------------------------------------------------------- typedef & struct ---
typedef struct {
    float x, y, z;
    float r, g, b, a;
} vertex_t;

struct GlyphsFreetypePrivate {
    // ------------------------------------------------------- global variables ---
    text_buffer_t *text_buffer;
    markup_t markup;
};


GlyphsFreetype::
        GlyphsFreetype()
    :p(new GlyphsFreetypePrivate)
{
#ifndef FT_CONFIG_OPTION_SUBPIXEL_RENDERING
    fprintf(stderr,
            "This demo requires freetype to be compiled "
            "with subpixel rendering.\n");
    exit( EXIT_FAILURE) ;
#endif

#ifndef __APPLE__
    glewExperimental = GL_TRUE;
    GLenum err = glewInit();
    if (GLEW_OK != err)
    {
        /* Problem: glewInit failed, something is seriously wrong. */
        fprintf( stderr, "Error: %s\n", glewGetErrorString(err) );
        exit( EXIT_FAILURE );
    }
    fprintf( stderr, "Using GLEW %s\n", glewGetString(GLEW_VERSION) );
#endif

    int program = ShaderResource::loadGLSLProgram(
                ":/shaders/freetype-gl-text.vert",
                ":/shaders/freetype-gl-text.frag");

    p->text_buffer = text_buffer_new_with_program( LCD_FILTERING_ON, program );
    vec4 black  = {{0.0, 0.0, 0.0, 1.0}};
    p->text_buffer->base_color = black;

    vec4 none   = {{1.0, 1.0, 1.0, 0.0}};
    p->markup.family = 0;
    p->markup.size    = 16.0;
    p->markup.bold    = 0;
    p->markup.italic  = 0;
    p->markup.rise    = 0.0;
    p->markup.spacing = 0.0;
    p->markup.gamma   = 1.0;
    p->markup.foreground_color    = black;
    p->markup.background_color    = none;
    p->markup.underline           = 0;
    p->markup.underline_color     = black;
    p->markup.overline            = 0;
    p->markup.overline_color      = black;
    p->markup.strikethrough       = 0;
    p->markup.strikethrough_color = black;
    QResource qr(":/fonts/Vera.ttf");
    p->markup.font = texture_font_new_from_memory(
                p->text_buffer->manager->atlas,
                p->markup.size,
                qr.data (), qr.size ());
    if (!p->markup.font)
    {
        Log("glyphsfreetype: failed to load font");
        text_buffer_delete ( p->text_buffer );
        p.reset ();
        return;
    }
}


GlyphsFreetype::
        ~GlyphsFreetype()
{
    text_buffer_delete ( p->text_buffer );
}


void GlyphsFreetype::
        drawGlyphs( const glProjection& gl_projection,  const std::vector<GlyphData>& glyphdata )
{
    if (!p)
        return;

    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd (gl_projection.projection.v ());
    glMatrixMode(GL_MODELVIEW);

    glEnable( GL_BLEND );
    glBlendFunc( GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA );
    buildGlyphs(glyphdata);

    glUseProgram( p->text_buffer->shader );
    {
        glUniformMatrix4fv( glGetUniformLocation( p->text_buffer->shader, "model" ),
                            1, 0, tmatrix<4,float>::identity ().v ());
        glUniformMatrix4fv( glGetUniformLocation( p->text_buffer->shader, "view" ),
                            1, 0, tmatrix<4,float>::identity ().v ());
        glUniformMatrix4fv( glGetUniformLocation( p->text_buffer->shader, "projection" ),
                            1, 0, GLmatrixf(gl_projection.projection).v ());
        text_buffer_render( p->text_buffer );
    }
}


void GlyphsFreetype::
        buildGlyphs(const std::vector<GlyphData>& glyphdata)
{
    text_buffer_clear( p->text_buffer );

    typedef tvector<4,GLfloat> v4f;
    std::vector<v4f> quad_v( 4*glyphdata.size () );
    v4f* quad = &quad_v[0];
    int quad_i = 0;

    float f = p->markup.size/1.5; // texture detail, how big a glyph is in the
                               // coordinate system of 'text_buffer_add_text'

    for (const GlyphData& g : glyphdata) {
        std::wstring text;
        std::copy(g.text.begin (), g.text.end (), std::inserter(text,text.end()));

        p->markup.spacing = g.letter_spacing; // pen->x += glyph->advance_x * (1.0 + markup->spacing);

        vec2 pen; pen.x = pen.y = 0;
        text_buffer_add_text( p->text_buffer, &pen, &p->markup, text.c_str (), text.size () );
        double w = pen.x;

        matrixd modelview = g.modelview;
        modelview *= matrixd::scale (1/f,1/f,1.);
        modelview *= matrixd::translate (g.margin*f - w*g.align_x, -g.align_y*f, 0);

        text_buffer_transform_last_line( p->text_buffer, tmatrix<4,float>(modelview).v () );

        float z = .3*f;
        float q = .3*f;
        quad[quad_i++] = modelview * v4f(0 - z, 0 - q, 0, 1);
        quad[quad_i++] = modelview * v4f(w + z, 0 - q, 0, 1);
        quad[quad_i++] = modelview * v4f(w + z, f + q, 0, 1);
        quad[quad_i++] = modelview * v4f(0 - z, f + q, 0, 1);
    }

    glLoadIdentity ();
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(4, GL_FLOAT, 0, quad);
    glColor4f(1,1,1,0.5);
    glDrawArrays(GL_QUADS, 0, quad_i);
    glDisableClientState(GL_VERTEX_ARRAY);
}


} // namespace Render
} // namespace Heightmap

#endif // USE_FREETYPE_GL
