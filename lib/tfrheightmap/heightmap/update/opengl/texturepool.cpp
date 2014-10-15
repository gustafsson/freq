#include "texturepool.h"
#include "gl.h"
#include "GlException.h"
#include "heightmap/render/blocktextures.h"
#include "log.h"
#include "datastorage.h"

namespace Heightmap {
namespace Update {
namespace OpenGL {


// compare to Render::BlockTextures::setupTexture
void setupTextureFloat32(unsigned name, unsigned w, unsigned h)
{
    glBindTexture(GL_TEXTURE_2D, name);
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

    // Compatible with GlFrameBuffer
#ifdef GL_ES_VERSION_2_0
    // https://www.khronos.org/registry/gles/extensions/EXT/EXT_texture_storage.txt
    GlException_SAFE_CALL( glTexStorage2DEXT ( GL_TEXTURE_2D, 1, GL_R32F_EXT, w, h));
#else
    GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D, 0, GL_R32F, w, h, 0, GL_RED, GL_FLOAT, 0) );
#endif
    glBindTexture(GL_TEXTURE_2D, 0);
}


void setupTexture(unsigned name, unsigned w, unsigned h, bool f32)
{
    if (f32)
        setupTextureFloat32(name,w,h);
    else
        Render::BlockTextures::setupTexture(name,w,h,false);

    // if it should use mipmaps, they should downsample along the w axis only. i.e GL_TEXTURE_1D_ARRAY
    //glGenerateMipmap (GL_TEXTURE_1D_ARRAY);
}


TexturePool::
        TexturePool(int width, int height, FloatSize format)
    :
      width_(width),
      height_(height),
      format_(format)
{
}


size_t TexturePool::
        size()
{
    return pool.size ();
}


void TexturePool::
        resize(size_t n)
{
    pool.resize (n);

    for (GlTexture::ptr& t : pool)
        if (!t)
            t = newTexture();

    Log("texturepool: %dx%dx%d using %s") % width_ % height_ % n
            % DataStorageVoid::getMemorySizeText (width_*height_*(format_/8)*n);
}


GlTexture::ptr TexturePool::
        get1()
{
    // get1 will increase the size of the pool if no textures were available
    for (GlTexture::ptr& t : pool)
        if (t.unique ())
            return t;

    pool.push_back (newTexture());
    return pool.back ();
}


GlTexture::ptr TexturePool::
        newTexture()
{
    GLuint t;
    glGenTextures (1, &t);
    setupTexture(t, width_, height_, format_ == Float32);
    bool adopt = true; // GlTexture does glDeleteTextures
    return GlTexture::ptr(new GlTexture(t, width_, height_, adopt));
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
