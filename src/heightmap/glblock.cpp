#include "gl.h"

#include "glblock.h"

// sonicawe
#include "sawe/nonblockingmessagebox.h"

// gpumisc
#include "vbo.h"
#include "demangle.h"
#include "GlException.h"
#include "cpumemorystorage.h"
#include "computationkernel.h"
#include "TaskTimer.h"

// std
#include <stdio.h>

// Qt
#include <QResource>


//#define TIME_COMPILESHADER
#define TIME_COMPILESHADER if(0)

//#define TIME_GLBLOCK
#define TIME_GLBLOCK if(0)


using namespace std;


namespace Heightmap {

// Helpers based on Cuda SDK sample, ocean FFT
// TODO check license terms of the Cuda SDK

// Attach shader to a program
string attachShader(GLuint prg, GLenum type, const char *name)
{
    stringstream result;

    TIME_COMPILESHADER TaskTimer tt("Compiling shader %s", name);
    try {
        GLuint shader;
        FILE * fp=0;
        int size, compiled;
        char * src;

        shader = glCreateShader(type);

        QResource qr(name);
        if (!qr.isValid())
            throw ios::failure(string("Couldn't find shader resource ") + name);
        if ( 0 == qr.size())
            throw ios::failure(string("Shader resource empty ") + name);

        size = qr.size();
        src = (char*)qr.data();
        glShaderSource(shader, 1, (const char**)&src, (const GLint*)&size);
        glCompileShader(shader);
        glGetShaderiv(shader, GL_COMPILE_STATUS, (GLint*)&compiled);

        if (fp) free(src);

        char shaderInfoLog[2048];
        glGetShaderInfoLog(shader, sizeof(shaderInfoLog), 0, shaderInfoLog);

        bool showShaderLog = !compiled;
#ifdef _DEBUG
        QString qshaderInfoLog(shaderInfoLog);
        showShaderLog |= 0 != qshaderInfoLog.contains("fail", Qt::CaseInsensitive);
        showShaderLog |= 0 != qshaderInfoLog.contains("warning", Qt::CaseInsensitive);
        showShaderLog |= strlen(shaderInfoLog)>0;
#endif

        if (showShaderLog)
        {
            result << "Failed to compile shader '" << name << "'"  << endl
                   << shaderInfoLog << endl;
        }

        if (compiled)
        {
            glAttachShader(prg, shader);
        }

        glDeleteShader(shader);

    } catch (const exception &x) {
        TIME_COMPILESHADER TaskInfo("Failed, throwing %s", vartype(x).c_str());
        throw;
    }

    return result.str();
}

// Create shader program from vertex shader and fragment shader files
GLuint loadGLSLProgram(const char *vertFileName, const char *fragFileName)
{
    GLint linked;
    GLuint program;
    stringstream resultLog;

    program = glCreateProgram();
    try {
        resultLog << attachShader(program, GL_VERTEX_SHADER, vertFileName);
        resultLog << attachShader(program, GL_FRAGMENT_SHADER, fragFileName);

        glLinkProgram(program);
        glGetProgramiv(program, GL_LINK_STATUS, &linked);

        char programInfoLog[2048];
        glGetProgramInfoLog(program, sizeof(programInfoLog), 0, programInfoLog);
        TaskTimer tt("Linking vertex shader %s with fragment shader %s\n%s",
                 vertFileName, fragFileName, programInfoLog);

        bool showProgramLog = !linked;
#ifdef _DEBUG
        QString qprogramInfoLog(programInfoLog);
        showProgramLog |= 0 != qprogramInfoLog.contains("fail", Qt::CaseInsensitive);
        showProgramLog |= 0 != qprogramInfoLog.contains("warning", Qt::CaseInsensitive);
        showProgramLog |= strlen(programInfoLog)>0;
#endif

        if (showProgramLog)
        {
            stringstream log;
            log     << "Failed to link fragment shader (" << fragFileName << ") "
                    << "with vertex shader (" << vertFileName << ")" << endl
                    << programInfoLog << endl
                    << resultLog.str();

            TaskInfo("Couldn't properly setup graphics\n%s", log.str().c_str());

            Sawe::NonblockingMessageBox::show(
                    QMessageBox::Critical,
                    "Couldn't properly setup graphics",
                    "Sonic AWE couldn't properly setup required graphics. "
                    "Please file this as a bug report to help us fix this. "
                    "See more info in 'Help->Report a bug'",

                    log.str().c_str() );
        }

        glUseProgram(program);

        GLenum glError = glGetError();
        if (GL_NO_ERROR != glError)
        {
            TaskInfo("glUseProgram failed %s", gluErrorString(glError));
            program = 0;
        }

        glUseProgram( 0 );

    } catch (...) {
        glDeleteProgram(program);
        throw;
    }
    return program;
}


GlBlock::
GlBlock( BlockLayout block_size, float width, float height )
:   block_size_( block_size ),
//    _read_only_array_resource( 0 ),
//    _read_only_array( 0 ),
    _tex_height(0),
    _tex_height_nearest(0),
    _world_width(width),
    _world_height(height)
{
    TIME_GLBLOCK TaskTimer tt("GlBlock()");

    // TODO read up on OpenGL interop in CUDA 3.0, cudaGLRegisterBufferObject is old, like CUDA 1.0 or something ;)
}


GlBlock::
~GlBlock()
{
    boost::shared_ptr<TaskTimer> tt;
    TIME_GLBLOCK tt.reset( new TaskTimer ("~GlBlock() _height=%u", _height?*_height:0u ));

    // no point in doing a proper unmapping when it might fail and the textures
    // that would recieve the updates are deleted right after anyways
    // unmap();
    if (_mapped_height)
    {
        EXCEPTION_ASSERT( _mapped_height.unique() );
        _mapped_height.reset();
        TIME_GLBLOCK TaskInfo("_mapped_height.reset()");
    }

    if (tt) tt->partlyDone();

    _height.reset();
    _mesh.reset();
    if (tt) tt->partlyDone();

    delete_texture();
    if (tt) tt->partlyDone();
}


void GlBlock::
        reset( float width, float height )
{
    _world_width = width;
    _world_height = height;
}


DataStorageSize GlBlock::
        heightSize() const
{
    return DataStorageSize(
                block_size_.texels_per_row (),
                block_size_.texels_per_column ());
}


GlBlock::pHeightReadOnlyCpu GlBlock::
        heightReadOnlyCpu()
{
    if (_read_only_cpu) return _read_only_cpu;

    if (_mapped_height)
    {
        // Transfer from Cuda instead of OpenGL if it can already be found in Cuda memory
        _read_only_cpu.reset(new DataStorage<float>( _mapped_height->data->size() ));
        *_read_only_cpu = *_mapped_height->data;
        return _read_only_cpu;
    }

    createHeightVbo();

    glBindBuffer(_height->vbo_type(), *_height);
    float *cpu_height = (float *) glMapBuffer(_height->vbo_type(), GL_READ_ONLY);

    _read_only_cpu.reset( new DataStorage<float>( *CpuMemoryStorage::BorrowPtr( heightSize(), cpu_height ) ));

    glUnmapBuffer(_height->vbo_type());
    glBindBuffer(_height->vbo_type(), 0);

    return _read_only_cpu;
}


/*GlBlock::HeightReadOnlyArray GlBlock::
        heightReadOnlyArray()
{
    if (_read_only_array) return _read_only_array;

    if (0==_read_only_array_resource)
        CudaException_SAFE_CALL( cudaGraphicsGLRegisterImage(
                &_read_only_array_resource,
                _tex_height,
                GL_TEXTURE_2D,
                cudaGraphicsMapFlagsReadOnly) );

    CudaException_SAFE_CALL( cudaGraphicsMapResources(1, &_read_only_array_resource, 0 ) );
    CudaException_SAFE_CALL( cudaGraphicsSubResourceGetMappedArray(
            &_read_only_array,
            _read_only_array_resource,
            0, 0));

    return _read_only_array;
}*/


void GlBlock::
        createHeightVbo()
{
    if (!_height)
    {
        TIME_GLBLOCK TaskTimer tt("Heightmap, creating vbo");
        unsigned elems = block_size_.texels_per_row ()*block_size_.texels_per_column ();
        // PIXEL_UNPACK_BUFFER, to be used with glTexSubImage2D
        _height.reset( new Vbo(elems*sizeof(float), GL_PIXEL_UNPACK_BUFFER, GL_STATIC_DRAW) );
    }
}


GlBlock::pHeight GlBlock::
        height()
{
    if (_mapped_height) return _mapped_height;

    createHeightVbo();

    TIME_GLBLOCK TaskTimer tt("Heightmap OpenGL->Cuda, vbo=%u", (unsigned)*_height);

    _mapped_height.reset( new MappedVbo<float>(_height, heightSize() ));

    TIME_GLBLOCK ComputationSynchronize();

    return _mapped_height;
}


bool GlBlock::
        has_texture()
{
    if (_tex_height_nearest)
        EXCEPTION_ASSERT(_tex_height);

    return _tex_height;
}


void GlBlock::
        delete_texture()
{
    unmap();

    if (_tex_height)
    {
        TIME_GLBLOCK TaskInfo("Deleting tex_height=%u", _tex_height);
        glDeleteTextures(1, &_tex_height);
        _tex_height = 0;
    }
    if (_tex_height_nearest)
    {
        TIME_GLBLOCK TaskInfo("Deleting _tex_height_nearest=%u", _tex_height_nearest);
        glDeleteTextures(1, &_tex_height_nearest);
        _tex_height_nearest = 0;
    }
}


bool GlBlock::
        create_texture( GlBlock::HeightMode heightMode )
{
    int w = block_size_.texels_per_row ();
    int h = block_size_.texels_per_column ();
    static bool hasTextureFloat = 0 != strstr( (const char*)glGetString(GL_EXTENSIONS), "GL_ARB_texture_float" );

    if (0==_tex_height)
    {
        glGenTextures(1, &_tex_height);
        glBindTexture(GL_TEXTURE_2D, _tex_height);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        //glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
        //glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        //glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);

        GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D,0,hasTextureFloat?GL_LUMINANCE32F_ARB:GL_LUMINANCE,w, h,0, hasTextureFloat?GL_LUMINANCE:GL_RED, GL_FLOAT, 0) );

        glBindTexture(GL_TEXTURE_2D, 0);

        TIME_GLBLOCK TaskInfo("Created tex_height=%u", _tex_height);
    }

    if (heightMode==HeightMode_VertexTexture && 0==_tex_height_nearest)
    {
        glGenTextures(1, &_tex_height_nearest);
        glBindTexture(GL_TEXTURE_2D, _tex_height_nearest);

        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MIN_FILTER,GL_NEAREST);
        glTexParameteri(GL_TEXTURE_2D,GL_TEXTURE_MAG_FILTER,GL_NEAREST);

        GlException_SAFE_CALL( glTexImage2D(GL_TEXTURE_2D,0,hasTextureFloat?GL_LUMINANCE32F_ARB:GL_LUMINANCE,w, h,0, hasTextureFloat?GL_LUMINANCE:GL_RED, GL_FLOAT, 0) );

        glBindTexture(GL_TEXTURE_2D, 0);

        TIME_GLBLOCK TaskInfo("Created tex_height_nearest=%d", _tex_height_nearest);

        return true;
    }

    if (heightMode==HeightMode_VertexBuffer && 0==_mesh)
    {
        _mesh.reset( new Vbo( (w+2)*(h+2)*4*sizeof(float), GL_ARRAY_BUFFER, GL_STATIC_DRAW) );

        return true;
    }

    return false;
}


void GlBlock::
        update_texture( GlBlock::HeightMode heightMode )
{
    bool got_new_height_data = 0==_tex_height || (bool)_mapped_height;
    bool got_new_vertex_data = create_texture( heightMode );

    got_new_vertex_data |= got_new_height_data && HeightMode_Flat != heightMode;
    if (!got_new_height_data && !got_new_vertex_data)
        return;

    int w = block_size_.texels_per_row ();
    int h = block_size_.texels_per_column ();
    static bool hasTextureFloat = 0 != strstr( (const char*)glGetString(GL_EXTENSIONS), "GL_ARB_texture_float" );

    if (!hasTextureFloat)
        glPixelTransferf( GL_RED_SCALE, 0.1f );

    if (got_new_height_data)
    {
        unmap();

        TIME_GLBLOCK TaskTimer tt("Updating heightmap texture=%u, vbo=%u", _tex_height, (unsigned)*_height);

        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, *_height );
        glBindTexture(GL_TEXTURE_2D, _tex_height);

        // See method comment in header file if you get an error on this row
        GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, w, h, hasTextureFloat?GL_LUMINANCE:GL_RED, GL_FLOAT, 0) );

        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
    }

    if (got_new_vertex_data && HeightMode_VertexTexture == heightMode)
    {
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, *_height );
        glBindTexture(GL_TEXTURE_2D, _tex_height_nearest);

        // See method comment in header file if you get an error on this row
        GlException_SAFE_CALL( glTexSubImage2D(GL_TEXTURE_2D,0,0,0, w, h, hasTextureFloat?GL_LUMINANCE:GL_RED, GL_FLOAT, 0) );

        glBindTexture(GL_TEXTURE_2D, 0);
        glBindBuffer( GL_PIXEL_UNPACK_BUFFER, 0);
    }

    if (got_new_vertex_data && HeightMode_VertexBuffer == heightMode)
    {
        glBindBuffer (_height->vbo_type(), *_height);
        float *cpu_height = (float *) glMapBuffer (_height->vbo_type(), GL_READ_ONLY);

        GlException_CHECK_ERROR ();

        // update vertex buffer to store mesh vertices
        int y1 = 0, x1 = 0, y2 = h, x2 = w;

        // edge dropout to eliminate visible glitches
        if (w>2) x1--, x2++;
        if (h>2) y1--, y2++;

        glBindBuffer (_mesh->vbo_type(), *_mesh);
        float *pos = (float *) glMapBuffer (_mesh->vbo_type(), GL_WRITE_ONLY);

        GlException_CHECK_ERROR ();

        for(int y=y1; y<y2; y++) {
            float v = y / (float) (h-1);
            int ry = min (h-1, max (0, y));
            for(int x=x1; x<x2; x++) {
                int rx = min (w-1, max (0, x));
                float H = cpu_height[rx + ry*w];

                float u = x / (float) (w-1);
                *pos++ = u;
                *pos++ = H;
                *pos++ = v;
                *pos++ = 1.0f;
            }
        }

        GlException_SAFE_CALL( glUnmapBuffer(_mesh->vbo_type()) );
        glBindBuffer(_mesh->vbo_type(), 0);

        GlException_SAFE_CALL( glUnmapBuffer(_height->vbo_type()) );
        glBindBuffer(_height->vbo_type(), 0);
    }

    if (!hasTextureFloat)
        glPixelTransferf( GL_RED_SCALE, 1.0f );

    TIME_GLBLOCK ComputationSynchronize();
}


void GlBlock::
        unmap()
{
    if (_read_only_cpu)
        _read_only_cpu.reset();

    /*if (_read_only_array)
    {
        cudaGraphicsUnmapResources( 1, &_read_only_array_resource );
        _read_only_array = 0;
        cudaGraphicsUnregisterResource( _read_only_array_resource );
        _read_only_array_resource = 0;
    }*/

    if (_mapped_height)
    {
        TIME_GLBLOCK TaskTimer tt("Heightmap Cuda->OpenGL, height=%u", (unsigned)*_height);
        TIME_GLBLOCK ComputationCheckError();

        EXCEPTION_ASSERT( _mapped_height.unique() );
        EXCEPTION_ASSERT( _mapped_height->data.unique() );

        _mapped_height.reset();

        TIME_GLBLOCK ComputationSynchronize();
    }
}


void GlBlock::
        draw( unsigned vbo_size, GlBlock::HeightMode withHeightMap )
{
    if (!_height)
    {
        TIME_GLBLOCK TaskInfo("Skipping rendering of block without data");
        return;
    }

    TIME_GLBLOCK ComputationCheckError();
    TIME_GLBLOCK GlException_CHECK_ERROR();

    update_texture( withHeightMap );

    switch(withHeightMap)
    {
    case HeightMode_Flat:
        break;

    case HeightMode_VertexTexture:
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, _tex_height_nearest );
        glActiveTexture(GL_TEXTURE0);
        break;

    case HeightMode_VertexBuffer:
        glBindBuffer(GL_ARRAY_BUFFER, *_mesh);
        glVertexPointer(4, GL_FLOAT, 0, 0);
        glEnableClientState(GL_VERTEX_ARRAY);
        break;
    }
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
    switch(withHeightMap)
    {
    case HeightMode_Flat:
        break;

    case HeightMode_VertexTexture:
        glActiveTexture(GL_TEXTURE1);
        glBindTexture(GL_TEXTURE_2D, 0);
        glActiveTexture(GL_TEXTURE0);
        break;

    case HeightMode_VertexBuffer:
        glBindBuffer(GL_ARRAY_BUFFER, 0);
        break;
    }

    TIME_GLBLOCK ComputationCheckError();
    TIME_GLBLOCK GlException_CHECK_ERROR();
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
    if (_tex_height) s += sizeof(float); // OpenGL texture
    if (_tex_height_nearest) s += sizeof(float); // OpenGL texture

    // _mapped_slope and _slope are temporary and only lives in the scope of update_texture

    return s;
}


} // namespace Heightmap
