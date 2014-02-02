#ifndef RESAMPLETEXTURE_H
#define RESAMPLETEXTURE_H

#include "GlTexture.h"
#include "glframebuffer.h"

/**
 * @brief The ResampleTexture class should paint a texture on top of another texture.
 */
class ResampleTexture: boost::noncopyable
{
public:
    struct Area
    {
        const float x1, y1, x2, y2;

        Area(float x1, float y1, float x2, float y2);
    };

    ResampleTexture(GlTexture* dest, Area destarea);
    ~ResampleTexture();

    void clear(float r=0, float g=0, float b=0, float a=0);
    void operator ()(GlTexture* source, Area area);
    void drawColoredArea(Area area, float r, float g=0, float b=0, float a=0);

private:
    GlFrameBuffer fbo;
    unsigned vbo;
    GlTexture* dest;
    Area destarea;

public:
    static void test();
};

#endif // RESAMPLETEXTURE_H
