#include "glblock.h"

// Heightmap namespace
#include "collection.h"
#include "renderer.h"
#include "slope.cu.h"

// gpumisc
#include <vbo.h>
#include <demangle.h>
#include <GlException.h>
#include <CudaException.h>

// cuda
#include <cuda_runtime.h>

// std
#include <stdio.h>

// Qt
#include <QResource>


#define TIME_COMPILESHADER
//#define TIME_COMPILESHADER if(0)

//#define TIME_GLBLOCK
#define TIME_GLBLOCK if(0)

namespace Heightmap {

// Helpers based on Cuda SDK sample, ocean FFT
// TODO check license terms of the Cuda SDK

// Attach shader to a program
void attachShader(GLuint prg, GLenum type, const char *name)
{
    TIME_COMPILESHADER TaskTimer tt("Compiling shader %s", name);
    try {
        GLuint shader;
        FILE * fp=0;
        int size, compiled;
        char * src;

        shader = glCreateShader(type);

        QResource qr(name);
        if (!qr.isValid())
            throw std::ios::failure(std::string("Couldn't find shader resource ") + name);
        if ( 0 == qr.size())
            throw std::ios::failure(std::string("Shader resource empty ") + name);

        size = qr.size();
        src = (char*)qr.data();
        glShaderSource(shader, 1, (const char**)&src, (const GLint*)&size);
        glCompileShader(shader);
        glGetShaderiv(shader, GL_COMPILE_STATUS, (GLint*)&compiled);

        if (fp) free(src);

        char log[2048];
        int len;

        glGetShaderInfoLog(shader, 2048, (GLsizei*)&len, log);

        if (!compiled) {
            glDeleteShader(shader);

            throw std::runtime_error(std::string("Couldn't compile shader ") + name + ". Shader log: \n" + log);
        }

        if (0<len) {
            TIME_COMPILESHADER TaskInfo("Shader log:\n%s", log);
        }

        glAttachShader(prg, shader);
        glDeleteShader(shader); // TODO why delete shader?
    } catch (const std::exception &x) {
        TIME_COMPILESHADER TaskInfo("Failed, throwing %s", vartype(x).c_str());
        throw;
    }
}

// Create shader program from vertex shader and fragment shader files
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName)
{
    GLint linked;
    GLuint program;

    program = glCreateProgram();
    try {
        attachShader(program, GL_VERTEX_SHADER, vertFileName);
        attachShader(program, GL_FRAGMENT_SHADER, fragFileName);

        glLinkProgram(program);
        glGetProgramiv(program, GL_LINK_STATUS, &linked);
        if (!linked) {
            char temp[256];
            glGetProgramInfoLog(program, 256, 0, temp);
            throw std::runtime_error(std::string("Failed to link shader program with vertex shader ")
                                     + vertFileName + " and fragment shader " + fragFileName
                                     + ". Program log:\n" + temp);
        }

    } catch (...) {
        glDeleteProgram(program);
        throw;
    }
    return program;
}

GlBlock::
GlBlock( Collection* collection, float width, float height )
:   _collection( collection ),
    _tex_height(0),
    _tex_slope(0),
    _world_width(width),
    _world_height(height),
    _got_new_height_data(false)
{
    TIME_GLBLOCK TaskTimer tt("GlBlock()");

    // TODO read up on OpenGL interop in CUDA 3.0, cudaGLRegisterBufferObject is old, like CUDA 1.0 or something ;)
}

GlBlock::
~GlBlock()
{
    boost::scoped_ptr<TaskTimer> tt;
    TIME_GLBLOCK tt.reset( new TaskTimer ("~GlBlock() tex_height=%u, tex_slope=%u", _tex_height, _tex_slope));

    // no point in doing a proper unmapping when it might fail and the textures
    // that would recieve the updates are deleted right after anyways
    // unmap();
    if (_mapped_height)
    {
        BOOST_ASSERT( _mapped_height.unique() );
        _mapped_height.reset();
        TIME_GLBLOCK TaskInfo("_mapped_height.reset()");
    }
    if (_mapped_slope)
    {
        BOOST_ASSERT( _mapped_slope.unique() );
        _mapped_slope.reset();
        TIME_GLBLOCK TaskInfo("_mapped_slope.reset()");
    }

    if (tt) tt->partlyDone();

    _height.reset();
    if (tt) tt->partlyDone();

    _slope.reset();
    if (tt) tt->partlyDone();

    delete_texture();
    if (tt) tt->partlyDone();
}

GlBlock::pHeight GlBlock::
height()
{
    if (_mapped_height) return _mapped_height;

    if (!_height)
    {
        TIME_GLBLOCK TaskTimer tt("Heightmap, creating vbo");
        unsigned elems = _collection->samples_per_block()*_collection->scales_per_block();
        _height.reset( new Vbo(elems*sizeof(float), GL_PIXEL_UNPACK_BUFFER) );
        _height->registerWithCuda();
    }

    TIME_GLBLOCK TaskTimer tt("Heightmap OpenGL->Cuda, vbo=%u", (unsigned)*_height);

    _mapped_height.reset( new MappedVbo<float>(_height, make_cudaExtent(
            _collection->samples_per_block(),
            _collection->scales_per_block(), 1)));

    TIME_GLBLOCK CudaException_ThreadSynchronize();

    return _mapped_height;
}

GlBlock::pSlope GlBlock::
slope()
{
    if (_mapped_slope) return _mapped_slope;

    if (!_slope)
    {
        TIME_GLBLOCK TaskTimer tt("Slope, creating vbo");

        unsigned elems = _collection->samples_per_block()*_collection->scales_per_block();
        _slope.reset( new Vbo(elems*sizeof(float2), GL_PIXEL_UNPACK_BUFFER) );
        _slope->registerWithCuda();
    }

    TIME_GLBLOCK TaskTimer tt("Slope OpenGL->Cuda, vbo=%u", (unsigned)*_slope);

    _mapped_slope.reset( new MappedVbo<float2>(_slope, make_cudaExtent(
            _collection->samples_per_block(),
            _collection->scales_per_block(), 1)));

    TIME_GLBLOCK CudaException_ThreadSynchronize();

    return _mapped_slope;
}


void GlBlock::
        delete_texture()
{
    if (_tex_height)
    {
        glDeleteTextures(1, &_tex_height);
        _tex_height = 0;
    }
    if (_tex_slope)
    {
        glDeleteTextures(1, &_tex_slope);
        _tex_slope = 0;
    }
}


void GlBlock::
        create_texture(bool create_slope)
{
    if (0==_tex_height)
    {
        glGenTextures(1, &_tex_height);
        glBindTexture(GL_TEXTURE_2D, _tex_height);
        unsigned w = _collection->samples_per_block();
        unsigned h = _collection->scales_per_block();

        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        //glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        //glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0); // no mipmaps

        glTexImage2D(GL_TEXTURE_2D,0,GL_LUMINANCE32F_ARB,w, h,0, GL_RED, GL_FLOAT, 0);

        glBindTexture(GL_TEXTURE_2D, 0);

        _got_new_height_data = true;

        TIME_GLBLOCK TaskInfo("Created tex_height=%u", _tex_height);
    }

    if (create_slope && 0==_tex_slope)
    {
        glGenTextures(1, &_tex_slope);
        glBindTexture(GL_TEXTURE_2D, _tex_slope);
        unsigned w = _collection->samples_per_block();
        unsigned h = _collection->scales_per_block();

        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
        //glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        //glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_BASE_LEVEL, 0);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAX_LEVEL, 0); // no mipmaps

        glTexImage2D(GL_TEXTURE_2D,0,GL_LUMINANCE_ALPHA32F_ARB,w, h,0, GL_LUMINANCE_ALPHA, GL_FLOAT, 0);

        glBindTexture(GL_TEXTURE_2D, 0);

        _got_new_height_data = true;

        TIME_GLBLOCK TaskInfo("Created tex_slope=%d", _tex_slope);
    }

}


void GlBlock::
        update_texture( bool create_slope )
{
    bool got_new_slope_data = create_slope && 0==_tex_slope;
    create_texture( create_slope );

    _got_new_height_data |= (bool)_mapped_height;
    got_new_slope_data |= _got_new_height_data;

    if (create_slope)
    {
        // Need a slope

#ifdef _MSC_VER
        if (false)
        {
            // TODO just calling 'slope()->data->getCudaGlobal()' here crashes the graphics driver in windows
            // slope()->data->getCudaGlobal()
#else
        if (got_new_slope_data)
        {
            // Slope needs to be updated (before unmap())
#endif

            computeSlope(0);

            {
                TIME_GLBLOCK TaskTimer tt("Slope Cuda->OpenGL, vbo=%u", (unsigned)*_slope);
                TIME_GLBLOCK CudaException_CHECK_ERROR();

                BOOST_ASSERT( _mapped_slope.unique() );

                _mapped_slope.reset();
                TIME_GLBLOCK CudaException_ThreadSynchronize();
            }

            TIME_GLBLOCK TaskTimer tt("Updating slope texture=%u, vbo=%u", _tex_slope, (unsigned)*_slope);

            unsigned texture_width = _collection->samples_per_block();
            unsigned texture_height = _collection->scales_per_block();

            glBindBuffer( GL_PIXEL_UNPACK_BUFFER, *_slope );
            glBindTexture(GL_TEXTURE_2D, _tex_slope);

            GlException_CHECK_ERROR();

            glTexSubImage2D(GL_TEXTURE_2D,0,0,0, texture_width, texture_height, GL_LUMINANCE_ALPHA, GL_FLOAT, 0);

            GlException_CHECK_ERROR(); // See method comment in header file if you get an error on this row

            glBindTexture(GL_TEXTURE_2D, 0);
            glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);

            TIME_GLBLOCK CudaException_CHECK_ERROR();

            _slope.reset();

            if (!_got_new_height_data)
                _mapped_height.reset();

            TIME_GLBLOCK CudaException_ThreadSynchronize();
        }
    }

    if (_got_new_height_data)
    {
        unmap();

        TIME_GLBLOCK TaskTimer tt("Updating heightmap texture=%u, vbo=%u", _tex_height, (unsigned)*_height);

        unsigned texture_width = _collection->samples_per_block();
        unsigned texture_height = _collection->scales_per_block();

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, *_height );
        glBindTexture(GL_TEXTURE_2D, _tex_height);
        glPixelTransferf(GL_RED_SCALE, 4.f);

        GlException_CHECK_ERROR();
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0, texture_width, texture_height, GL_RED, GL_FLOAT, 0);
        GlException_CHECK_ERROR(); // See method comment in header file if you get an error on this row

        glPixelTransferf(GL_RED_SCALE, 1.0f);

        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);

        TIME_GLBLOCK CudaException_ThreadSynchronize();

        _got_new_height_data = false;
    }
}


void GlBlock::
        unmap()
{
    if (_mapped_height)
    {
        TIME_GLBLOCK TaskTimer tt("Heightmap Cuda->OpenGL, height=%u", (unsigned)*_height);
        TIME_GLBLOCK CudaException_CHECK_ERROR();

        BOOST_ASSERT( _mapped_height.unique() );

        _mapped_height.reset();

        TIME_GLBLOCK CudaException_ThreadSynchronize();

        _got_new_height_data = true;
    }
}

void GlBlock::
        draw( unsigned vbo_size )
{    
    TIME_GLBLOCK CudaException_CHECK_ERROR();
    TIME_GLBLOCK GlException_CHECK_ERROR();

    update_texture( true );

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _tex_slope);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _tex_height);

    const bool wireFrame = false;
    const bool drawPoints = false;

    glColor4f(1.0, 1.0, 1.0, 1.0);
    if (drawPoints) {
        glDrawArrays(GL_POINTS, 0, vbo_size);
    } else if (wireFrame) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
            glDrawElements(GL_TRIANGLE_STRIP, vbo_size, GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else {
        glDrawElements(GL_TRIANGLE_STRIP, vbo_size, GL_UNSIGNED_INT, 0);
    }
    glBindTexture(GL_TEXTURE_2D, 0);

    TIME_GLBLOCK CudaException_CHECK_ERROR();
    TIME_GLBLOCK GlException_CHECK_ERROR();
}

void GlBlock::
        draw_flat( )
{
    update_texture( false );

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0); // no slope texture for drawing a flat rectangle
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _tex_height);

    glDisable(GL_BLEND);
    glDisable(GL_COLOR_MATERIAL);
    glDisable(GL_LIGHTING);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

    glDisable(GL_CULL_FACE);
    glColor4f(1.0, 1.0, 1.0, 1.0);
    glBegin( GL_TRIANGLE_STRIP );
        glTexCoord2f(0,0);    glVertex3f(0,0,0);
        glTexCoord2f(0,1);    glVertex3f(1,0,0);
        glTexCoord2f(1,0);    glVertex3f(0,0,1);
        glTexCoord2f(1,1);    glVertex3f(1,0,1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
}

//static int clamp(int val, int max) {
//    if (val<0) return 0;
//    if (val>max) return max;
//    return val;
//}

//static void setWavelengthColor( float wavelengthScalar ) {
//    const float spectrum[][3] = {
//        /* white background */
//        { 1, 1, 1 },
//        { 0, 0, 1 },
//        { 0, 1, 1 },
//        { 0, 1, 0 },
//        { 1, 1, 0 },
//        { 1, 0, 1 },
//        { 1, 0, 0 }};
//        /* black background
//        { 0, 0, 0 },
//        { 1, 0, 1 },
//        { 0, 0, 1 },
//        { 0, 1, 1 },
//        { 0, 1, 0 },
//        { 1, 1, 0 },
//        { 1, 0, 0 }}; */

//    unsigned count = sizeof(spectrum)/sizeof(spectrum[0]);
//    float f = count*wavelengthScalar;
//    unsigned i = clamp(f, count-1);
//    unsigned j = clamp(f+1, count-1);
//    float t = f-i;

//    GLfloat rgb[] = {  spectrum[i][0]*(1-t) + spectrum[j][0]*t,
//                       spectrum[i][1]*(1-t) + spectrum[j][1]*t,
//                       spectrum[i][2]*(1-t) + spectrum[j][2]*t
//                   };
//    glColor3fv( rgb );
//}

//void GlBlock::
//draw_directMode( )
//{
//    pHeight block = height();
//    cudaExtent n = block->data->getNumberOfElements();
//    const float* data = block->data->getCpuMemory();

//    float ifs = 1./(n.width-3);
//    float depthScale = 1.f/(n.height-3);

//    glEnable(GL_NORMALIZE);
//    for (unsigned fi=1; fi<n.height-2; fi++)
//    {
//        glBegin(GL_TRIANGLE_STRIP);
//            float v[3][4] = {{0}}; // vertex buffer needed to compute normals

//            for (unsigned t=0; t<n.width; t++)
//            {
//                for (unsigned dt=0; dt<2; dt++)
//                    for (unsigned df=0; df<4; df++)
//                        v[dt][df] = v[dt+1][df];

//                for (int df=-1; df<3; df++)
//                    v[2][df+1] = data[ t + (fi + df)*n.width ];

//                if (2>t) // building v
//                    continue;

//                // Add two vertices to the triangle strip.
//                for (int j=0; j<2; j++)
//                {
//                    setWavelengthColor( v[1][j+1] );
//                    float dt=(v[2][j+1]-v[0][j+1]);
//                    float df=(v[1][j+2]-v[1][j+0]);
//                    glNormal3f( -dt, 2, -df ); // normalized by OpenGL
//                    glVertex3f( ifs*(t-2), v[1][j+1], (fi-1+j)*depthScale);
//                }
//            }
//        glEnd(); // GL_TRIANGLE_STRIP
//    }
//    glDisable(GL_NORMALIZE);
//}


unsigned GlBlock::
        allocated_bytes_per_element()
{
    unsigned s = 0;
    if (_height) s += sizeof(float); // OpenGL VBO
    if (_mapped_height) s += sizeof(float); // Cuda device memory
    if (_tex_height) s += 2*sizeof(float); // OpenGL texture, 2 times the size for mipmaps
    if (_tex_slope) s += 2*sizeof(float2); // OpenGL texture, 2 times the size for mipmaps

    // _mapped_slope and _slope are temporary and only lives in the scope of update_texture

    return s;
}


void GlBlock::
        computeSlope( unsigned /*cuda_stream */)
{
    TIME_GLBLOCK TaskTimer tt("Slope computeSlope");

    ::cudaCalculateSlopeKernel( height()->data->getCudaGlobal(),
                                slope()->data->getCudaGlobal(),
                                _world_width, _world_height );

    TIME_GLBLOCK CudaException_ThreadSynchronize();
}

} // namespace Heightmap
