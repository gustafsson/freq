#include "glyphsglut.h"


// glut
#ifndef __APPLE__
#   include <GL/glut.h>
#else
# ifndef GL_ES_VERSION_2_0
#   include <GLUT/glut.h>
# endif
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
        drawGlyphs( const glProjection& gl_projection, const std::vector<GlyphData>& glyphs)
{
#ifndef GL_ES_VERSION_2_0
    glMatrixMode(GL_PROJECTION);
    glLoadMatrixd (gl_projection.projection.v ());
    glMatrixMode(GL_MODELVIEW);

    typedef tvector<2,GLfloat> GLvector2F;
    std::vector<GLvector2F> quad(4);

    for (const GlyphData& g : glyphs) {
        double w = g.margin*100.;
        double letter_spacing = g.letter_spacing*100.;
        const char* a = g.text.c_str ();
        for (const char*c=a;*c!=0; c++)
        {
            if (c!=a)
                w+=letter_spacing;
            w+=glutStrokeWidth( GLUT_STROKE_ROMAN, *c );
        }

        matrixd modelview = g.modelview;
        modelview *= matrixd::scale (0.01,0.01,1.);
        modelview *= matrixd::translate (-w*g.align_x,-g.align_y*100.,0);

        glLoadMatrixd (modelview.v ());

        float z = 10;
        float q = 20;
        glEnableClientState(GL_VERTEX_ARRAY);
        quad[0] = GLvector2F(0 - z, 0 - q);
        quad[1] = GLvector2F(w + z, 0 - q);
        quad[2] = GLvector2F(0 - z, 100 + q);
        quad[3] = GLvector2F(w + z, 100 + q);
        glVertexPointer(2, GL_FLOAT, 0, &quad[0]);
        glColor4f(1,1,1,0.5);
        glDrawArrays(GL_TRIANGLE_STRIP, 0, quad.size());
        glDisableClientState(GL_VERTEX_ARRAY);

        glColor4f(0,0,0,0.8);
        for (const char*c=a;*c!=0; c++)
        {
            glutStrokeCharacter(GLUT_STROKE_ROMAN, *c);
            modelview *= matrixd::translate (letter_spacing + glutStrokeWidth( GLUT_STROKE_ROMAN, *c ),0,0);
            glLoadMatrixd (modelview.v ());
        }
    }
#endif // GL_ES_VERSION_2_0
}


} // namespace Render
} // namespace Heightmap

