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

#include "heightmap-collection.h"
#include "heightmap-renderer.h"

#include <stdio.h>
#include <QResource>

//#define TIME_SHADER
#define TIME_SHADER if(0)

namespace Heightmap {

// Helpers based on Cuda SDK sample, ocean FFT
// TODO check license terms of the Cuda SDK

// Attach shader to a program
void attachShader(GLuint prg, GLenum type, const char *name)
{
    TIME_SHADER TaskTimer tt("Compiling shader %s", name);
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
			TIME_SHADER TaskTimer("Shader log:\n%s", log).suppressTiming();
        }


        glAttachShader(prg, shader);
        glDeleteShader(shader);
    } catch (const std::exception &x) {
#ifndef __APPLE__
        TIME_SHADER TaskTimer("Failed, throwing %s", demangle(typeid(x).name()).c_str()).suppressTiming();
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
    _height( new Vbo(collection->samples_per_block()*collection->scales_per_block()*sizeof(float)) ),
    _slope( new Vbo(collection->samples_per_block()*collection->scales_per_block()*sizeof(float2)) )
{
    //_renderer->setSize(renderer->spectrogram()->samples_per_block(), renderer->spectrogram()->scales_per_block());
    cudaGLRegisterBufferObject(*_height);
    cudaGLRegisterBufferObject(*_slope);
}

GlBlock::
~GlBlock()
{
    unmap();

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
    if (_mapped_height) {
        TaskTimer tt(TaskTimer::LogVerbose, "Heightmap Cuda->OpenGL");
        _mapped_height.reset();
    }
    if (_mapped_slope) {
        TaskTimer tt(TaskTimer::LogVerbose, "Gradient Cuda->OpenGL");
        _mapped_slope.reset();
    }
}

void GlBlock::
        draw()
{
    unsigned meshW = _collection->samples_per_block();
    unsigned meshH = _collection->scales_per_block();

    unmap();

    glBindBuffer(GL_ARRAY_BUFFER, *_height);
    glClientActiveTexture(GL_TEXTURE0);
    glTexCoordPointer(1, GL_FLOAT, 0, 0);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, *_slope);
    glClientActiveTexture(GL_TEXTURE1);
    glTexCoordPointer(2, GL_FLOAT, 0, 0);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);




    bool wireFrame = false;
    bool drawPoints = false;

    glColor3f(1.0, 1.0, 1.0);
    if (drawPoints) {
        glDrawArrays(GL_POINTS, 0, meshW * meshH);
    } else {
        glPolygonMode(GL_FRONT_AND_BACK, wireFrame ? GL_LINE : GL_FILL);
            glDrawElements(GL_TRIANGLE_STRIP, ((meshW*2)+2)*(meshH-1), GL_UNSIGNED_INT, 0);
        glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
    }

    glClientActiveTexture(GL_TEXTURE0);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glClientActiveTexture(GL_TEXTURE1);
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
}

int clamp(int val, int max) {
    if (val<0) return 0;
    if (val>max) return max;
    return val;
}

void setWavelengthColor( float wavelengthScalar ) {
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
