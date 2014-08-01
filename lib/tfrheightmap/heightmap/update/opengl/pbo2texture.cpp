#include "pbo2texture.h"
#include "neat_math.h"

#include "heightmap/render/shaderresource.h"
#include "tfr/chunk.h"

#include "gl.h"
#include "GlException.h"
#include "tasktimer.h"

#include <QResource>

//#define INFO
#define INFO if(0)

static void initUpdateShaders() {
    Q_INIT_RESOURCE(updateshaders);
}

namespace Heightmap {
namespace Update {
namespace OpenGL {

int gl_max_texture_size() {
    static int v = 0;
    if (0 == v)
    {
        glGetIntegerv (GL_MAX_TEXTURE_SIZE, &v);
        // GL_MAX_TEXTURE_SIZE-1 seems to be the largest allowed texture in practice...
        v = lpo2s (v); // power of 2 is slightly faster
    }
    return v;
}


Shader::Shader(GLuint program)
    :
      program(program),
      normalization_location_(-1),
      amplitude_axis_location_(-1),
      modelViewProjectionMatrix_location_(-1),
      data_size_loc_(-1),
      tex_size_loc_(-1)
{
    EXCEPTION_ASSERT( program );

    GlException_SAFE_CALL( glUseProgram(program) );
    int mytex = glGetUniformLocation(program, "mytex");
    GlException_CHECK_ERROR();
    data_size_loc_ = glGetUniformLocation(program, "data_size");
    GlException_CHECK_ERROR();
    tex_size_loc_ = glGetUniformLocation(program, "tex_size");
    GlException_CHECK_ERROR();
    normalization_location_ = glGetUniformLocation(program, "normalization");
    GlException_CHECK_ERROR();
    amplitude_axis_location_ = glGetUniformLocation(program, "amplitude_axis");
    GlException_CHECK_ERROR();
    modelViewProjectionMatrix_location_ = glGetUniformLocation (program, "qt_ModelViewProjectionMatrix");
    GlException_SAFE_CALL( glUniform1i(mytex, 0) ); // mytex corresponds to GL_TEXTURE0
    GlException_SAFE_CALL( glUseProgram(0) );
}


Shader::~Shader()
{
    if (program)
        glDeleteProgram(program);
}


void Shader::Shader::
        setParams(int data_width, int data_height, int tex_width, int tex_height,
               float normalization_factor, int amplitude_axis, const glProjection& M)
{
    EXCEPTION_ASSERT( program );

    GlException_SAFE_CALL( glUseProgram(program) );
    if ( 0 <= data_size_loc_)
        glUniform2f(data_size_loc_, data_width, data_height);
    if ( 0 <= tex_size_loc_)
        glUniform2f(tex_size_loc_, tex_width, tex_height);
    if ( 0 <= normalization_location_)
        glUniform1f(normalization_location_, normalization_factor);
    if ( 0 <= amplitude_axis_location_)
        glUniform1i(amplitude_axis_location_, amplitude_axis);
    if ( 0 <= modelViewProjectionMatrix_location_)
        glUniformMatrix4fv (modelViewProjectionMatrix_location_, 1, false, (M.projection * M.modelview).v ());
    GlException_SAFE_CALL( glUseProgram(0) );
}


Shaders::Shaders():
    load_shaders_([](){initUpdateShaders();return 0;}()),
    chunktoblock_shader_(ShaderResource::loadGLSLProgram(":/shaders/chunktoblock.vert", ":/shaders/chunktoblock.frag")),
    chunktoblock_maxwidth_shader_(ShaderResource::loadGLSLProgram(":/shaders/chunktoblock.vert", ":/shaders/chunktoblock_maxwidth.frag")),
    chunktoblock_maxheight_shader_(ShaderResource::loadGLSLProgram(":/shaders/chunktoblock.vert", ":/shaders/chunktoblock_maxheight.frag"))
{
}


ShaderTexture::ShaderTexture(Shaders &shaders)
    :
      shaders_(shaders)
{
}


void ShaderTexture::
        prepareShader (int data_width, int data_height, float* p)
{
    prepareShader(data_width, data_height, 0, p);
}


void ShaderTexture::
        prepareShader (int data_width, int data_height, unsigned chunk_pbo_)
{
    prepareShader(data_width, data_height, chunk_pbo_, 0);
}


GlTexture& ShaderTexture::
        getTexture ()
{
    EXCEPTION_ASSERT(chunk_texture_);

    return *chunk_texture_;
}

unsigned ShaderTexture::
        getProgram (float normalization_factor, int amplitude_axis, const glProjection& M)
{
    shader_->setParams (data_width, data_height, tex_width, tex_height, normalization_factor, amplitude_axis, M);
    return shader_->program;
}


void ShaderTexture::
        prepareShader (int data_width, int data_height, unsigned chunk_pbo_, float* p)
{
    this->data_width = data_width;
    this->data_height = data_height;

    // Either through chunk_vbo_ or p
    EXCEPTION_ASSERT((chunk_pbo_!=0) == (p==0));

    INFO TaskTimer tt(boost::format("ChunkToBlockDegenerateTexture::prepTexture %u x %u <> %d")
                      % data_width % data_height % gl_max_texture_size());

    if (data_width <= gl_max_texture_size() && data_height <= gl_max_texture_size())
      {
        // No need for degeneracy
        shader_ = &shaders_.chunktoblock_shader_;

        // But a power of 2 might help
        if (0 == "do a power of 2")
        {
            // Could use a power of 2, but it doesn't really help
            tex_width = spo2g (data_width-1);
            tex_height = spo2g (data_height-1);
        }
        else
        {
            tex_width = data_width;
            tex_height = data_height;
        }


        INFO TaskTimer tt("glTexSubImage2D %d x %d (1)", tex_width, tex_height);
        chunk_texture_.reset (new GlTexture( tex_width, tex_height, GL_RED, GL_RED, GL_FLOAT, 0));
        GlTexture::ScopeBinding texObjBinding = chunk_texture_->getScopeBinding();
        GlException_SAFE_CALL( glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_) );

        glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, data_width, data_height, GL_RED, GL_FLOAT, p);

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      }
    else if (data_width > gl_max_texture_size())
      {
        // Degeneracy. Make the size a multiple of data_height.
        shader_ = &shaders_.chunktoblock_maxwidth_shader_;

        tex_width = gl_max_texture_size();
        int s = int_div_ceil (data_width, (tex_width-1));
        tex_height = data_height * s;

        int N = data_width*data_height;
        int M = tex_height*tex_width;
        EXCEPTION_ASSERT_LESS_OR_EQUAL(N,M);

        // Would run out of OpenGL memory if this happens...
        EXCEPTION_ASSERT_LESS_OR_EQUAL(tex_height, gl_max_texture_size());

        INFO TaskTimer tt("glTexSubImage2D %d x %d (2)", tex_width, tex_height);
        chunk_texture_.reset (new GlTexture( tex_width, tex_height, GL_RED, GL_RED, GL_FLOAT, 0));
        GlTexture::ScopeBinding texObjBinding = chunk_texture_->getScopeBinding();
        GlException_SAFE_CALL( glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_) );

        if (0 == chunk_pbo_)
          {
            // GL_UNPACK_ROW_LENGTH with a very large data_width doesn't play well with PBO
            TaskTimer tt("GL_UNPACK_ROW_LENGTH");
            GlException_SAFE_CALL( glPixelStorei(GL_UNPACK_ROW_LENGTH, data_width) );

            for (int i=0; i<s; ++i)
              {
                int w = std::min(tex_width, data_width - (tex_width-1)*i);
                int y = i*data_height;
                int n = i*(tex_width-1);
                glTexSubImage2D (GL_TEXTURE_2D, 0, 0, y, w, data_height, GL_RED, GL_FLOAT, p + n);
              }

            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
          }
        else
          {
            for (int h=0; h<data_height; ++h)
              {
                for (int i=0; i<s; ++i)
                  {
                    int w = std::min(tex_width, data_width - (tex_width-1)*i);
                    int y = h + i*data_height;
                    int n = i*(tex_width-1) + h*data_width;
                    glTexSubImage2D (GL_TEXTURE_2D, 0, 0, y, w, 1, GL_RED, GL_FLOAT, p + n);
                  }
              }
          }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

        // Clear remaining
        int i = s-1;
        int w = std::min(tex_width, data_width - (tex_width-1)*i);
        int leftcols = tex_width - w;

        if (leftcols > 0)
          {
            int leftdata = leftcols*data_height;
            std::vector<float> zeros(leftdata);
            glTexSubImage2D (GL_TEXTURE_2D, 0, w, i*data_height, leftcols, data_height, GL_RED, GL_FLOAT, &zeros[0]);
          }
      }
    else
      {
        shader_ = &shaders_.chunktoblock_maxheight_shader_;

        tex_height = gl_max_texture_size();
        int s = int_div_ceil (data_height, (tex_height-1));
        tex_width = data_width * s;

        // Out of open gl memory if this happens...
        EXCEPTION_ASSERT_LESS_OR_EQUAL(tex_width, gl_max_texture_size());

        INFO TaskTimer tt("glTexSubImage2D %d x %d (3)", tex_width, tex_height);
        chunk_texture_.reset (new GlTexture( tex_width, tex_height, GL_RED, GL_RED, GL_FLOAT, 0));
        GlTexture::ScopeBinding texObjBinding = chunk_texture_->getScopeBinding();
        GlException_SAFE_CALL( glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_) );

        for (int i=0; i<s; ++i)
          {
            int h = std::min(tex_height, data_height - (tex_height-1)*i);
            int w = data_width;
            int x = i*data_width;
            int n = data_width*i*(tex_height-1);
            glTexSubImage2D (GL_TEXTURE_2D, 0, x, 0, w, h, GL_RED, GL_FLOAT, p + n);
          }

        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
      }

    GlException_CHECK_ERROR();
}


Pbo2Texture::Pbo2Texture(Shaders& shaders, Tfr::pChunk chunk, int chunk_pbo)
    :
      shader_(shaders)
{
    bool transpose = chunk->order == Tfr::Chunk::Order_column_major;
    int data_width  = transpose ? chunk->nScales ()  : chunk->nSamples ();
    int data_height = transpose ? chunk->nSamples () : chunk->nScales ();

    shader_.prepareShader (data_width, data_height, chunk_pbo);
}


Pbo2Texture::ScopeMap Pbo2Texture::
        map (float normalization_factor, int amplitude_axis, const glProjection& M, int &vertex_attrib, int &tex_attrib)
{
    Pbo2Texture::ScopeMap r;
    unsigned program = shader_.getProgram (normalization_factor, amplitude_axis, M);
    vertex_attrib = glGetAttribLocation (program, "qt_Vertex");
    tex_attrib = glGetAttribLocation (program, "qt_MultiTexCoord0");
    glUseProgram(program);
    shader_.getTexture ().bindTexture2D ();

    return r;
}


Pbo2Texture::ScopeMap::
        ScopeMap()
{

}


Pbo2Texture::ScopeMap::
        ~ScopeMap()
{
    GlException_SAFE_CALL( glBindTexture( GL_TEXTURE_2D, 0) );
    GlException_SAFE_CALL( glDisable(GL_TEXTURE_2D) );
    glUseProgram(0);
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
