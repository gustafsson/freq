#if defined(USE_GLUT) && defined(LEGACY_OPENGL)

#include "glyphsglut.h"


// glut
#ifndef __APPLE__
#   include <GL/glut.h>
#else
#   include <GLUT/glut.h>
#endif


namespace Heightmap {
namespace Render {

GlyphsGlut::GlyphsGlut()
{
    // Using glut for drawing fonts, so glutInit must be called.
    static int c=0;
    if (0==c)
    {
        // run glutinit once per process
#ifdef _WIN32
        c = 1;
        char* dummy="dummy\0";
        glutInit(&c,&dummy);
#elif !defined(__APPLE__)
        glutInit(&c,0);
        c = 1;
#endif
    }
}


void GlyphsGlut::
        drawGlyphs( const glProjection& gl_projection, const std::vector<GlyphData>& glyphdata)
{
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd (gl_projection.projection.v ());
    glMatrixMode(GL_MODELVIEW);

    typedef tvector<4,GLfloat> v4f;
    v4f quad[4];
    int quad_i = 0;

    const float f = 100; // texture detail, how big a glyph is in the
                         // coordinate system of 'text_buffer_add_text'

    for (const GlyphData& g : glyphdata) {
        double w = 0;
        double spacing = g.letter_spacing + 0.1;
        const char* a = g.text.c_str ();
        for (const char*c=a;*c!=0; c++)
        {
            double sw = glutStrokeWidth( GLUT_STROKE_ROMAN, *c );
            if (c==a || c[1]==0)
                w += sw*(1+spacing*0.5);
            else
                w += sw*(1+spacing);
        }

        matrixd modelview = g.modelview;
        modelview *= matrixd::scale (1/f,1/f,1.);
        modelview *= matrixd::translate (g.margin*f - w*g.align_x, -g.align_y*f, 0);

        float z = .3*f;
        float q = .3*f;
        quad[quad_i++] = modelview * v4f(0 - z, 0 - q, 0, 1);
        quad[quad_i++] = modelview * v4f(w + z, 0 - q, 0, 1);
        quad[quad_i++] = modelview * v4f(w + z, f + q, 0, 1);
        quad[quad_i++] = modelview * v4f(0 - z, f + q, 0, 1);

        glLoadIdentity ();
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(4, GL_FLOAT, 0, quad);
        glColor4f(1,1,1,0.5);
        glDrawArrays(GL_QUADS, 0, quad_i);
        glDisableClientState(GL_VERTEX_ARRAY);
        quad_i = 0;

        glColor4f(0,0,0,0.8);
        for (const char*c=a;*c!=0; c++)
        {
            double sw = glutStrokeWidth( GLUT_STROKE_ROMAN, *c );
            if (c!=a)
                modelview *= matrixd::translate (spacing*0.5*sw, 0, 0);
            glLoadMatrixd (modelview.v ());
            glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
            modelview *= matrixd::translate ((spacing*0.5 + 1)*sw, 0, 0);
        }
    }
}


} // namespace Render
} // namespace Heightmap

#else
int USE_GLUT_GlyphsGlut;
#endif // defined(USE_GLUT) && defined(LEGACY_OPENGL)
