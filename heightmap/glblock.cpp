// Must include glew.h berfore gl.h
#ifdef _MSC_VER
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#define VC_EXTRALEAN
#include <windows.h>
#endif
#ifndef __APPLE__
  #include <GL/glew.h>
#endif

#include <vbo.h>
#include <demangle.h>
#include <GlException.h>
#include <CudaException.h>

#include "heightmap/collection.h"
#include "heightmap/renderer.h"

#include <stdio.h>
#include <QResource>

//#define TIME_COMPILESHADER
#define TIME_COMPILESHADER if(0)

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
                        TIME_COMPILESHADER TaskTimer("Shader log:\n%s", log).suppressTiming();
        }


        glAttachShader(prg, shader);
        glDeleteShader(shader); // TODO why delete shader?
    } catch (const std::exception &x) {
#ifndef __APPLE__
        TIME_COMPILESHADER TaskTimer("Failed, throwing %s", vartype(x).c_str()).suppressTiming();
#endif
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
GlBlock( Collection* collection )
:   _collection( collection ),
    _height( new Vbo(collection->samples_per_block()*collection->scales_per_block()*sizeof(float), GL_PIXEL_UNPACK_BUFFER) ),
    _slope( new Vbo(collection->samples_per_block()*collection->scales_per_block()*sizeof(float2), GL_PIXEL_UNPACK_BUFFER) ),
    _tex_height(0)
{
    TIME_GLBLOCK TaskTimer tt("GlBlock()");

    // TODO read up on OpenGL interop in CUDA 3.0, cudaGLRegisterBufferObject is old, like CUDA 1.0 or something ;)
    cudaGLRegisterBufferObject(*_height);
    cudaGLRegisterBufferObject(*_slope);

    glGenTextures(1, &_tex_height);
    glBindTexture(GL_TEXTURE_2D, _tex_height);
    unsigned w = collection->samples_per_block();
    unsigned h = collection->scales_per_block();
    float* p = new float[w*h];
    memset(p, 0, sizeof(float)*w*h);
    glTexImage2D(GL_TEXTURE_2D,0,GL_LUMINANCE32F_ARB,w, h,0, GL_RED, GL_FLOAT, p);
    delete[]p;

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );

    glGenTextures(1, &_tex_slope);
    glBindTexture(GL_TEXTURE_2D, _tex_slope);

    p = new float[w*h*2];
    memset(p, 0, sizeof(float)*w*h*2);
    glTexImage2D(GL_TEXTURE_2D,0,GL_LUMINANCE_ALPHA32F_ARB,w, h,0, GL_LUMINANCE_ALPHA, GL_FLOAT, 0);
    delete[]p;

    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE );
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE );

    glBindTexture(GL_TEXTURE_2D, 0);

    TIME_GLBLOCK TaskTimer("_tex_height=%u, _tex_slope=%d", _tex_height, _tex_slope).suppressTiming();
}

GlBlock::
~GlBlock()
{
    TIME_GLBLOCK TaskTimer tt("~GlBlock() _tex_height=%u",_tex_height);

    unmap();

    if (_tex_height)
    {
        glDeleteTextures(1, &_tex_height);
        _tex_height = 0;
    }

    cudaGLUnregisterBufferObject(*_height);
    cudaGLUnregisterBufferObject(*_slope);
}

GlBlock::pHeight GlBlock::
height()
{
    if (_mapped_height) return _mapped_height;
    return _mapped_height = pHeight(new MappedVbo<float>(_height, make_cudaExtent(
            _collection->samples_per_block(),
            _collection->scales_per_block(), 1)));
}

GlBlock::pSlope GlBlock::
slope()
{
    if (_mapped_slope) return _mapped_slope;
    return _mapped_slope = pSlope(new MappedVbo<float2>(_slope, make_cudaExtent(
            _collection->samples_per_block(),
            _collection->scales_per_block(), 1)));
}

void GlBlock::
        unmap()
{
    if (_mapped_height)
    {
        TIME_GLBLOCK TaskTimer tt("Heightmap Cuda->OpenGL");

        BOOST_ASSERT( _mapped_height.unique() );

        _mapped_height.reset();

        unsigned meshW = _collection->samples_per_block();
        unsigned meshH = _collection->scales_per_block();

        TIME_GLBLOCK TaskTimer("meshW=%u, meshH=%u, _tex_height=%u, *_height=%u", meshW, meshH, _tex_height, (unsigned)*_height).suppressTiming();
        glActiveTexture(GL_TEXTURE0);
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, *_height );
        glBindTexture(GL_TEXTURE_2D, _tex_height);
        glPixelTransferf(GL_RED_SCALE, 4.f);

        GlException_CHECK_ERROR();
        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,meshW, meshH,GL_RED, GL_FLOAT, 0);
        GlException_CHECK_ERROR(); // See method comment in header file if you get an error on this row

        glPixelTransferf(GL_RED_SCALE, 1.0f);

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
    }

    if (_mapped_slope)
    {
        TIME_GLBLOCK TaskTimer tt("Gradient Cuda->OpenGL");

        BOOST_ASSERT( _mapped_slope.unique() );

        _mapped_slope.reset();

        unsigned meshW = _collection->samples_per_block();
        unsigned meshH = _collection->scales_per_block();

        TIME_GLBLOCK TaskTimer("meshW=%u, meshH=%u, _tex_slope=%u, *_slope=%u", meshW, meshH, _tex_slope, (unsigned)*_slope).suppressTiming();
        glActiveTexture(GL_TEXTURE0);
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, *_slope );
        glBindTexture(GL_TEXTURE_2D, _tex_slope);
        //glPixelTransferf(GL_RED_SCALE, 1.0f);
        //glPixelTransferf(GL_ALPHA_SCALE, 1.0f);

        GlException_CHECK_ERROR();

        glTexSubImage2D(GL_TEXTURE_2D,0,0,0,meshW, meshH,GL_LUMINANCE_ALPHA, GL_FLOAT, 0);
        //glTexSubImage2D(GL_TEXTURE_2D,0,0,0,meshW, meshH,GL_RG, GL_FLOAT, 0);

        GlException_CHECK_ERROR(); // See method comment in header file if you get an error on this row

        //glPixelTransferf(GL_RED_SCALE, 1.0f);
        //glPixelTransferf(GL_ALPHA_SCALE, 1.0f);

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
    }
}

void GlBlock::
        draw()
{
    unmap();

    unsigned meshW = _collection->samples_per_block();
    unsigned meshH = _collection->scales_per_block();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, _tex_slope);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _tex_height);

    const bool wireFrame = false;
    const bool drawPoints = false;

    glColor3f(1.0, 1.0, 1.0);
    if (drawPoints) {
        glDrawArrays(GL_POINTS, 0, meshW * meshH);
    } else if (wireFrame) {
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE );
            glDrawElements(GL_TRIANGLE_STRIP, ((meshW*2)+3)*(meshH-1), GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    } else {
        glDrawElements(GL_TRIANGLE_STRIP, ((meshW*2)+3)*(meshH-1), GL_UNSIGNED_INT, 0);
    }
    glBindTexture(GL_TEXTURE_2D, 0);
}

void GlBlock::
        draw_flat( )
{
    unmap();

    glActiveTexture(GL_TEXTURE1);
    glBindTexture(GL_TEXTURE_2D, 0);
    glActiveTexture(GL_TEXTURE0);
    glBindTexture(GL_TEXTURE_2D, _tex_height);

    glBegin( GL_TRIANGLE_STRIP );
        glTexCoord2f(0.0,0.0);    glVertex3f(0,0,0);
        glTexCoord2f(0.0,1.0);    glVertex3f(0,0,1);
        glTexCoord2f(1.0,0.0);    glVertex3f(1,0,0);
        glTexCoord2f(1.0,1.0);    glVertex3f(1,0,1);
    glEnd();

    glBindTexture(GL_TEXTURE_2D, 0);
}

static int clamp(int val, int max) {
    if (val<0) return 0;
    if (val>max) return max;
    return val;
}

static void setWavelengthColor( float wavelengthScalar ) {
    const float spectrum[][3] = {
        /* white background */
        { 1, 1, 1 },
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 0, 1, 0 },
        { 1, 1, 0 },
        { 1, 0, 1 },
        { 1, 0, 0 }};
        /* black background
        { 0, 0, 0 },
        { 1, 0, 1 },
        { 0, 0, 1 },
        { 0, 1, 1 },
        { 0, 1, 0 },
        { 1, 1, 0 },
        { 1, 0, 0 }}; */

    unsigned count = sizeof(spectrum)/sizeof(spectrum[0]);
    float f = count*wavelengthScalar;
    unsigned i = clamp(f, count-1);
    unsigned j = clamp(f+1, count-1);
    float t = f-i;

    GLfloat rgb[] = {  spectrum[i][0]*(1-t) + spectrum[j][0]*t,
                       spectrum[i][1]*(1-t) + spectrum[j][1]*t,
                       spectrum[i][2]*(1-t) + spectrum[j][2]*t
                   };
    glColor3fv( rgb );
}

void GlBlock::
draw_directMode( )
{
    pHeight block = height();
    cudaExtent n = block->data->getNumberOfElements();
    const float* data = block->data->getCpuMemory();

    float ifs = 1./(n.width-3);
    float depthScale = 1.f/(n.height-3);

    glEnable(GL_NORMALIZE);
    for (unsigned fi=1; fi<n.height-2; fi++)
    {
        glBegin(GL_TRIANGLE_STRIP);
            float v[3][4] = {{0}}; // vertex buffer needed to compute normals

            for (unsigned t=0; t<n.width; t++)
            {
                for (unsigned dt=0; dt<2; dt++)
                    for (unsigned df=0; df<4; df++)
                        v[dt][df] = v[dt+1][df];

                for (int df=-1; df<3; df++)
                    v[2][df+1] = data[ t + (fi + df)*n.width ];

                if (2>t) // building v
                    continue;

                // Add two vertices to the triangle strip.
                for (int j=0; j<2; j++)
                {
                    setWavelengthColor( v[1][j+1] );
                    float dt=(v[2][j+1]-v[0][j+1]);
                    float df=(v[1][j+2]-v[1][j+0]);
                    glNormal3f( -dt, 2, -df ); // normalized by OpenGL
                    glVertex3f( ifs*(t-2), v[1][j+1], (fi-1+j)*depthScale);
                }
            }
        glEnd(); // GL_TRIANGLE_STRIP
    }
    glDisable(GL_NORMALIZE);
}

} // namespace Heightmap
