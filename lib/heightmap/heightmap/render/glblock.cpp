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
GlBlock( GlTexture::ptr tex )
:   tex_( tex ),
    _tex_height( tex->getOpenGlTextureId () )
{
    INFO TaskTimer tt("GlBlock()");

    // TODO read up on OpenGL interop in CUDA 3.0, cudaGLRegisterBufferObject is old, like CUDA 1.0 or something ;)
}


GlTexture::ptr GlBlock::
        glTexture()
{
    return tex_;
}


bool GlBlock::
        has_texture() const
{
    return _tex_height;
}




} // namespace Render
} // namespace Heightmap
