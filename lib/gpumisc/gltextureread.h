#ifndef GLTEXTUREREAD_H
#define GLTEXTUREREAD_H

#include "datastorage.h"

#ifndef GL_RGBA
#define GL_RGBA 0x1908
#else
#if GL_RGBA != 0x1908
#error GL_RGBA is different from 0x1908
#endif
#endif

/**
 * @brief The GlTextureRead class should read the contents of an OpenGL texture.
 */
class GlTextureRead
{
public:
    GlTextureRead(int texture);

    /**
     * @brief read
     * @param level
     * @param format See glGetTexImage
     * @return
     */
    DataStorage<float>::ptr readFloat(int level=0, int format=GL_RGBA);
    DataStorage<unsigned char>::ptr readByte(int level=0, int format=GL_RGBA);

private:
    int texture;

public:
    static void test();
};

#endif // GLTEXTUREREAD_H
