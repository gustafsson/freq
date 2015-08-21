#include "pbo2texture.h"
#include "neat_math.h"

#include "heightmap/render/shaderresource.h"
#include "heightmap/render/blocktextures.h"
#include "tfr/chunk.h"

#include "gl.h"
#include "GlException.h"
#include "tasktimer.h"
#include "log.h"

#include <QResource>

//#define INFO
#define INFO if(0)

static void initUpdateShaders() {
    Q_INIT_RESOURCE(updateshaders);
}

namespace Heightmap {
namespace Update {
namespace OpenGL {

Shader::Shader(ShaderPtr&& programp)
    :
      program(programp->programId()),
      normalization_location_(-1),
      amplitude_axis_location_(-1),
      modelViewProjectionMatrix_location_(-1),
      data_size_loc_(-1),
      tex_size_loc_(-1),
      programp_(std::move(programp))
{
    EXCEPTION_ASSERT( program );

    int mytex = -1;

    vertex_attrib_ = glGetAttribLocation (program, "qt_Vertex");
    tex_attrib_ = glGetAttribLocation (program, "qt_MultiTexCoord0");
    GlException_SAFE_CALL( mytex = glGetUniformLocation(program, "mytex") );
    GlException_SAFE_CALL( data_size_loc_ = glGetUniformLocation(program, "data_size") );
    GlException_SAFE_CALL( tex_size_loc_ = glGetUniformLocation(program, "tex_size") );
    GlException_SAFE_CALL( normalization_location_ = glGetUniformLocation(program, "normalization") );
    GlException_SAFE_CALL( amplitude_axis_location_ = glGetUniformLocation(program, "amplitude_axis") );
    GlException_SAFE_CALL( modelViewProjectionMatrix_location_ = glGetUniformLocation (program, "qt_ModelViewProjectionMatrix") );
    GlException_SAFE_CALL( glUseProgram(program) );
    GlException_SAFE_CALL( glUniform1i(mytex, 0) ); // mytex corresponds to GL_TEXTURE0
    GlException_SAFE_CALL( glUseProgram(0) );
}


Shader::~Shader()
{
}


void Shader::Shader::
        setParams(int data_width, int data_height, int tex_width, int tex_height,
               float normalization_factor, int amplitude_axis, const glProjection& M, int &vertex_attrib, int &tex_attrib)
{
    EXCEPTION_ASSERT( program );

    if ( 0 <= data_size_loc_ && (data_width != this->data_width || data_height != this->data_height))
        glUniform2f(data_size_loc_, this->data_width = data_width, this->data_height = data_height);
    if ( 0 <= tex_size_loc_ && (tex_width != this->tex_width || tex_height != this->tex_height))
        glUniform2f(tex_size_loc_, this->tex_width = tex_width, this->tex_height = tex_height);
    if ( 0 <= normalization_location_ && normalization_factor != this->normalization_factor)
        glUniform1f(normalization_location_, this->normalization_factor = normalization_factor);
    if ( 0 <= amplitude_axis_location_ && amplitude_axis != this->amplitude_axis)
        glUniform1i(amplitude_axis_location_, this->amplitude_axis = amplitude_axis);
    if ( 0 <= modelViewProjectionMatrix_location_ && (this->M.projection != M.projection || this->M.modelview != M.modelview))
    {
        this->M = M;
        glUniformMatrix4fv (modelViewProjectionMatrix_location_, 1, false, GLmatrixf(M.projection * M.modelview).v ());
    }

    vertex_attrib = vertex_attrib_;
    tex_attrib = tex_attrib_;
}


Shaders::Shaders():
    load_shaders_([](){initUpdateShaders();return 0;}()),
    chunktoblock_shader_(ShaderResource::loadGLSLProgram(":/shaders/chunktoblock.vert", ":/shaders/chunktoblock.frag")),
    chunktoblock_maxwidth_shader_(ShaderResource::loadGLSLProgram(":/shaders/chunktoblock.vert", ":/shaders/chunktoblock_maxwidth.frag")),
    chunktoblock_maxheight_shader_(ShaderResource::loadGLSLProgram(":/shaders/chunktoblock.vert", ":/shaders/chunktoblock_maxheight.frag"))
{
}


ShaderTexture::ShaderTexture(Shaders& shaders, GlTexture::ptr chunk_texture)
    :
      tex_width(chunk_texture->getWidth ()),
      tex_height(chunk_texture->getHeight ()),
      chunk_texture_(chunk_texture),
      shaders_(shaders)
{
}


void ShaderTexture::
        prepareShader (int data_width, int data_height, void* p, bool f32)
{
    prepareShader(data_width, data_height, 0, p, f32);
}


void ShaderTexture::
        prepareShader (int data_width, int data_height, unsigned chunk_pbo_, bool f32)
{
    prepareShader(data_width, data_height, chunk_pbo_, 0, f32);
}


GlTexture& ShaderTexture::
        getTexture () const
{
    EXCEPTION_ASSERT(chunk_texture_);

    return *chunk_texture_;
}


unsigned ShaderTexture::
        getProgram (float normalization_factor, int amplitude_axis, const glProjection& M, int &vertex_attrib, int &tex_attrib) const
{
    glUseProgram(shader_->program);
    shader_->setParams (data_width, data_height, tex_width, tex_height, normalization_factor, amplitude_axis, M, vertex_attrib, tex_attrib);
    return shader_->program;
}


void ShaderTexture::
        prepareShader (int data_width, int data_height, unsigned chunk_pbo_, void* p, bool f32)
{
    this->data_width = data_width;
    this->data_height = data_height;

    // Either through chunk_vbo_ or p, not both
    EXCEPTION_ASSERT((chunk_pbo_!=0) == (p==0));

#ifdef GL_ES_VERSION_2_0
    // PBO is not supported on OpenGL ES
    EXCEPTION_ASSERT_EQUALS(chunk_pbo_, 0u);

    // GL_R32F doesn't support linear interpolation on GL_ES, so use GL_R16F instead
    // GL_ES does support R32F, bot not linear interpolation
    //EXCEPTION_ASSERT_EQUALS(f32, false);
#endif

    GLenum format, type;
    char* pc = (char*)p;
    int element_size;
    int stride;

    if (f32)
    {
        #if defined(GL_ES_VERSION_2_0) && !defined(GL_ES_VERSION_3_0)
          format = GL_RED_EXT;
        #else
          format = GL_RED;
        #endif

        type = GL_FLOAT;
        element_size = sizeof(float);
        stride = data_width;
    }
    else
    {
        #ifdef GL_ES_VERSION_3_0
            EXCEPTION_ASSERTX(f32, "uploads with glTexSubImage2D only works with f32");
        #endif

        #if defined(GL_ES_VERSION_2_0) && !defined(GL_ES_VERSION_3_0)
          format = GL_RED_EXT;
          type = GL_HALF_FLOAT_OES;
        #else
          format = GL_RED;
          type = GL_HALF_FLOAT;
        #endif

        element_size = sizeof(int16_t);
        stride = (data_width+1)/2*2;
#ifdef LEGACY_OPENGL
        if (stride != data_width)
            glPixelStorei(GL_UNPACK_ROW_LENGTH,stride);
#endif
    }

    INFO TaskTimer tt(boost::format("ChunkToBlockDegenerateTexture::prepTexture data %u x %u <> tex %u x %u")
                      % data_width % data_height % tex_width % tex_height);

    int sw = int_div_ceil (data_width, (tex_width-1));
    int sh = int_div_ceil (data_height, (tex_height-1));

    if (data_width <= tex_width && data_height <= tex_height)
      {
        // No need for degeneracy
        shader_ = &shaders_.chunktoblock_shader_;

        INFO TaskTimer tt("glTexSubImage2D %d x %d (1)", tex_width, tex_height);
        chunk_texture_->bindTexture();

#ifdef LEGACY_OPENGL
        GlException_SAFE_CALL( glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_) );
#endif
        GlException_SAFE_CALL( glTexSubImage2D (GL_TEXTURE_2D, 0, 0, 0, data_width, data_height, format, type, p) );
#ifdef LEGACY_OPENGL
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#endif
      }
    else if (data_width > tex_width && sw*data_height <= tex_height)
      {
        // Degeneracy. Make the size a multiple of data_height.
        shader_ = &shaders_.chunktoblock_maxwidth_shader_;

        int N = data_width*data_height;
        int M = tex_height*tex_width;
        EXCEPTION_ASSERT_LESS_OR_EQUAL(N,M);

        INFO TaskTimer tt("glTexSubImage2D %d x %d (2)", tex_width, tex_height);
        chunk_texture_->bindTexture();
#ifdef LEGACY_OPENGL
        GlException_SAFE_CALL( glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_) );
#endif

#ifdef LEGACY_OPENGL
        if (0 == chunk_pbo_)
          {
            // GL_UNPACK_ROW_LENGTH with a very large data_width doesn't play well with PBO
            // and GL_UNPACK_ROW_LENGTH is not supported on OpenGL ES
            //TaskTimer tt("GL_UNPACK_ROW_LENGTH");
            GlException_SAFE_CALL( glPixelStorei(GL_UNPACK_ROW_LENGTH, stride) );

            for (int i=0; i<sw; ++i)
              {
                int w = std::min(tex_width, data_width - (tex_width-1)*i);
                int y = i*data_height;
                int n = i*(tex_width-1);
                glTexSubImage2D (GL_TEXTURE_2D, 0, 0, y, w, data_height, format, type, pc + n*element_size);
              }

            glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
          }
        else
#endif
          {
            for (int h=0; h<data_height; ++h)
              {
                for (int i=0; i<sw; ++i)
                  {
                    int w = std::min(tex_width, data_width - (tex_width-1)*i);
                    int y = h + i*data_height;
                    int n = i*(tex_width-1) + h*stride;
                    glTexSubImage2D (GL_TEXTURE_2D, 0, 0, y, w, 1, format, type, pc + n*element_size);
                  }
              }
          }

#ifdef LEGACY_OPENGL
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#endif
      }
    else if (data_height > tex_height && sh*data_width <= tex_width)
      {
        shader_ = &shaders_.chunktoblock_maxheight_shader_;

        int N = data_width*data_height;
        int M = tex_height*tex_width;
        EXCEPTION_ASSERT_LESS_OR_EQUAL(N,M);

        INFO TaskTimer tt("glTexSubImage2D %d x %d (3)", tex_width, tex_height);
        chunk_texture_->bindTexture();
#ifdef LEGACY_OPENGL
        GlException_SAFE_CALL( glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_) );
#endif

        for (int i=0; i<sh; ++i)
          {
            int h = std::min(tex_height, data_height - (tex_height-1)*i);
            int w = data_width;
            int x = i*data_width;
            int n = stride*i*(tex_height-1);
            glTexSubImage2D (GL_TEXTURE_2D, 0, x, 0, w, h, format, type, pc + n*element_size);
          }

#ifdef LEGACY_OPENGL
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
#endif
      }
    else
      {
        // Would run out of OpenGL memory if this happens...
        // Should run this mapping several times instead of creating a degenerate texture

        Log("pbo2texture: too large chunk to fit into a texture. Review doing multiple uploads instead\n"
            "data_width = %g\n"
            "data_height = %g\n"
            "tex_width = %g\n"
            "tex_height = %g\n"
            "sh = %g, sw = %g") % data_width % data_height % tex_width % tex_height % sh % sw;
      }

#ifdef LEGACY_OPENGL
    glPixelStorei(GL_UNPACK_ROW_LENGTH,0);
#endif

    GlException_CHECK_ERROR();
}


Pbo2Texture::Pbo2Texture(Shaders& shaders, GlTexture::ptr chunk_texture, Tfr::pChunk chunk, int chunk_pbo, bool f32)
    :
      shader_(shaders, chunk_texture)
{
    bool transpose = chunk->order == Tfr::Chunk::Order_column_major;
    int data_width  = transpose ? chunk->nScales ()  : chunk->nSamples ();
    int data_height = transpose ? chunk->nSamples () : chunk->nScales ();

    shader_.prepareShader (data_width, data_height, chunk_pbo, f32);
}


Pbo2Texture::Pbo2Texture(Shaders& shaders, GlTexture::ptr chunk_texture, Tfr::pChunk chunk, void* p, bool f32)
    :
      shader_(shaders, chunk_texture)
{
    bool transpose = chunk->order == Tfr::Chunk::Order_column_major;
    int data_width  = transpose ? chunk->nScales ()  : chunk->nSamples ();
    int data_height = transpose ? chunk->nSamples () : chunk->nScales ();

    shader_.prepareShader (data_width, data_height, p, f32);
}


Pbo2Texture::ScopeMap Pbo2Texture::
        map (float normalization_factor, int amplitude_axis, const glProjection& M, int &vertex_attrib, int &tex_attrib) const
{
    Pbo2Texture::ScopeMap r;
    shader_.getProgram (normalization_factor, amplitude_axis, M, vertex_attrib, tex_attrib);
    glBindTexture( GL_TEXTURE_2D, shader_.getTexture ().getOpenGlTextureId() );

    return r;
}


Pbo2Texture::ScopeMap::
        ScopeMap()
{

}


Pbo2Texture::ScopeMap::
        ~ScopeMap()
{
    glUseProgram(0);
}

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
