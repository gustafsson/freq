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
        float x1, y1, x2, y2;

        Area(float x1, float y1, float x2, float y2);
    };

    ResampleTexture(unsigned dest);
    ~ResampleTexture();

    GlFrameBuffer::ScopeBinding enable(Area destarea);

    void clear(float r=0, float g=0, float b=0, float a=0);
    void operator ()(GlTexture* source, Area area);
    void drawColoredArea(Area area, float r, float g=0, float b=0, float a=0);

private:
    std::shared_ptr<GlFrameBuffer> fbo;
    unsigned vbo;
    Area destarea;
    int width, height;

public:
    static void test();
    static void testInContext();
};

#endif // RESAMPLETEXTURE_H
