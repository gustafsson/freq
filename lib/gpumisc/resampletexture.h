#ifndef RESAMPLETEXTURE_H
#define RESAMPLETEXTURE_H

#include "glframebuffer.h"
#include "GlTexture.h"

/**
 * @brief The ResampleTexture class should paint a texture on top of another texture.
 */
class ResampleTexture: boost::noncopyable
{
#ifdef LEGACY_OPENGL
public:
    struct Area
    {
        float x1, y1, x2, y2;

        Area(float x1, float y1, float x2, float y2);
    };

    ResampleTexture(unsigned dest, int width, int height);
    ResampleTexture(const GlTexture& dest);
    ~ResampleTexture();

    GlFrameBuffer::ScopeBinding enable(Area destarea);

    void clear(float r=0, float g=0, float b=0, float a=0);
    void operator ()(GlTexture* source, Area area);
    void drawColoredArea(Area area, float r, float g=0, float b=0, float a=0);

private:
    GlFrameBuffer fbo;
    unsigned vbo;
    Area destarea;
#endif // LEGACY_OPENGL

public:
    static void test();
    static void testInContext();
};

#endif // RESAMPLETEXTURE_H
