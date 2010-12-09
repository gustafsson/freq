#ifndef HEIGHTMAPVBO_H
#define HEIGHTMAPVBO_H

#include "heightmap/collection.h"

#include <cuda_runtime.h>
#ifdef _MSC_VER // cuda_gl_interop.h includes gl.h which expects windows.h to be included on windows
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif
#include <cuda_gl_interop.h>
#include <GpuCpuData.h>
#include <stdio.h>
#include <mappedvbo.h>


namespace Heightmap {

GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName);

class GlBlock
{
public:
    GlBlock( Collection* collection );
    ~GlBlock();

    typedef boost::shared_ptr< MappedVbo<float> > pHeight;
    typedef boost::shared_ptr< MappedVbo<float2> > pSlope;

    pHeight height();
    pSlope slope();

    /**
        'unmap' releases copies of pHeight and pSlope held by GlBlock and
        updates the OpenGL textures that are used for rendering.

        It is an error for a client to call unmap while keeping an instance
        of pHeight or pSlope. Because if there is an instance of pHeight left.
        The Vbo is not unmapped from cuda, glBindBuffer doesn't do anything and
        glTexSubImage2D fails.
      */
    void unmap();

    void draw( );
    void draw_flat( );
    void draw_directMode( );
private:
    Collection* _collection;

    pVbo _height;
    pVbo _slope;

    pHeight _mapped_height;
    pSlope _mapped_slope;

    unsigned _tex_height;
    unsigned _tex_slope;

    bool _successfully_registered_height;
    bool _successfully_registered_slope;
};

typedef boost::shared_ptr<GlBlock> pGlBlock;

} // namespace Heightmap

#endif // HEIGHTMAPVBO_H

