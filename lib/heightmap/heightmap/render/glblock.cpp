#include "gl.h"

#include "glblock.h"

// gpumisc
#include "vbo.h"
#include "demangle.h"
#include "GlException.h"
#include "cpumemorystorage.h"
#include "computationkernel.h"
#include "tasktimer.h"

// std
#include <stdio.h>


//#define INFO
#define INFO if(0)


using namespace std;


namespace Heightmap {
namespace Render {

GlBlock::
GlBlock( BlockLayout block_size, float width, float height )
:   block_size_( block_size ),
    _tex_height(0),
    _world_width(width),
    _world_height(height)
{
    INFO TaskTimer tt("GlBlock()");

    // TODO read up on OpenGL interop in CUDA 3.0, cudaGLRegisterBufferObject is old, like CUDA 1.0 or something ;)
}


GlBlock::
~GlBlock()
{
    INFO TaskTimer tt("~GlBlock()");

    try {
        delete_texture();
    } catch (...) {
        TaskInfo(boost::format("!!! ~GlBlock: delete_texture failed\n%s") % boost::current_exception_diagnostic_information());
    }
}


void GlBlock::
        reset( float width, float height )
{
    _world_width = width;
    _world_height = height;
}


GlTexture::ptr GlBlock::
        glTexture()
{
    create_texture ();

    return GlTexture::ptr(new GlTexture(_tex_height));
}


void GlBlock::
        updateTexture( float*p, int n )
{
    create_texture ();

    static bool hasTextureFloat = 0 != strstr( (const char*)glGetString(GL_EXTENSIONS), "GL_ARB_texture_float" );

    int w = block_size_.texels_per_row ();
    int h = block_size_.texels_per_column ();

    EXCEPTION_ASSERT_EQUALS(n, block_size_.texels_per_block ());

    if (!hasTextureFloat)
        glPixelTransferf( GL_RED_SCALE, 0.1f );

    glBindTexture(GL_TEXTURE_2D, _tex_height);
    GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, w, h, hasTextureFloat?GL_LUMINANCE:GL_RED, GL_FLOAT, p) );
    glBindTexture(GL_TEXTURE_2D, 0);

    if (!hasTextureFloat)
        glPixelTransferf( GL_RED_SCALE, 1.0f );
}


bool GlBlock::
        has_texture() const
{
    return _tex_height;
}


void GlBlock::
        delete_texture()
{
    if (_tex_height)
    {
        INFO TaskInfo("Deleting tex_height=%u", _tex_height);
        glDeleteTextures(1, &_tex_height);
        _tex_height = 0;
    }
}


void GlBlock::
        create_texture()
{
    if (has_texture())
        return;

    int w = block_size_.texels_per_row ();
    int h = block_size_.texels_per_column ();
    static bool hasTextureFloat = 0 != strstr( (const char*)glGetString(GL_EXTENSIONS), "GL_ARB_texture_float" );

    glGenTextures(1, &_tex_height);
    glBindTexture(GL_TEXTURE_2D, _tex_height);

    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

    // Not compatible with GlFrameBuffer
    //GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D,0,hasTextureFloat?GL_LUMINANCE32F_ARB:GL_LUMINANCE,w, h,0, hasTextureFloat?GL_LUMINANCE:GL_RED, GL_FLOAT, 0) );

    // Compatible with GlFrameBuffer
    GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D,0,GL_RGBA,w, h,0, hasTextureFloat?GL_LUMINANCE:GL_RED, GL_FLOAT, 0) );

    glBindTexture(GL_TEXTURE_2D, 0);

    INFO TaskInfo("Created tex_height=%u", _tex_height);
}


void GlBlock::
        draw( unsigned vbo_size )
{
    if (!has_texture ())
        return;

    GlException_CHECK_ERROR();

    glBindTexture(GL_TEXTURE_2D, _tex_height);

    const bool wireFrame = false;
    const bool drawPoints = false;

    if (drawPoints) {
        glDrawArrays(GL_POINTS, 0, vbo_size);
    } else if (wireFrame) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
            glDrawElements(GL_TRIANGLE_STRIP, vbo_size, BLOCK_INDEX_TYPE, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else {
        glDrawElements(GL_TRIANGLE_STRIP, vbo_size, BLOCK_INDEX_TYPE, 0);
    }

    glBindTexture(GL_TEXTURE_2D, 0);

    GlException_CHECK_ERROR();
}


unsigned GlBlock::
        allocated_bytes_per_element() const
{
    unsigned s = 0;
    if (_tex_height) s += 4*sizeof(float); // OpenGL texture

    return s;
}

} // namespace Render
} // namespace Heightmap
