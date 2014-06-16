#include "chunktoblockdegeneratetexture.h"
#include "glblock.h"
#include "tfr/chunk.h"

#include "glframebuffer.h"
#include "tasktimer.h"
#include "gl.h"
#include "GlException.h"
#include "shaderresource.h"
#include "glPushContext.h"
#include "neat_math.h"

#include "tools/applicationerrorlogcontroller.h"

//#define INFO
#define INFO if(0)

namespace Heightmap {

struct vertex_format {
    float x, y, u, v;
};


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


void grabToTexture(GlTexture::ptr t)
{
    glBindTexture(GL_TEXTURE_2D, t->getOpenGlTextureId ());
    glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, t->getWidth (), t->getHeight ());
    glBindTexture(GL_TEXTURE_2D, 0);
}


BlockFbo::BlockFbo (pBlock block)
    :
      block(block),
      glblock(block->glblock)
{
    // Create new texture
//    glblock.reset(new GlBlock(block->block_layout(), block->getRegion().time (), block->getRegion().scale ()));
    fbo.reset (new GlFrameBuffer(glblock->glTexture ()->getOpenGlTextureId ()));

    // Copy from texture to own fbo
//    {
//        GlFrameBuffer fbo(block->glblock->glTexture ()->getOpenGlTextureId ());
//        fbo.bindFrameBuffer ();
//        grabToTexture(glblock->glTexture());
//        fbo.unbindFrameBuffer ();
//    }
}


BlockFbo::~BlockFbo()
{
    try {
//        fbo->bindFrameBuffer ();
//        grabToTexture(block->glblock->glTexture());
//        grabToTexture(block->glblock->glVertTexture());
//        fbo->unbindFrameBuffer ();

        // Discard previous glblock ... wrong thread ... could also grabToTexture into oldglblock
        fbo->bindFrameBuffer ();
        grabToTexture(glblock->glVertTexture());
        fbo->unbindFrameBuffer ();
//        glFinish ();
//        block->glblock = glblock;

        block->discard_new_block_data ();
    } catch (...) {
        Tools::ApplicationErrorLogController::registerException (boost::current_exception());
    }
}

void BlockFbo::
        begin ()
{
    GlException_CHECK_ERROR ();

    fbo->bindFrameBuffer ();
//        GlTexture::ScopeBinding texObjBinding = chunk_texture_->getScopeBinding();
//        GlException_SAFE_CALL( glDrawArrays(GL_TRIANGLE_STRIP, 0, nScales*2) );

    Region br = block->getRegion ();
    BlockLayout block_layout = block->block_layout ();

    // Juggle texture coordinates so that border texels are centered on the border
    float dt = br.time (), ds = br.scale ();
    br.a.time -= 0.5*dt / block_layout.texels_per_row ();
    br.b.time += 0.5*dt / block_layout.texels_per_row ();
    br.a.scale -= 0.5*ds / block_layout.texels_per_column ();
    br.b.scale += 0.5*ds / block_layout.texels_per_column ();

    // Setup matrices
    glViewport (0, 0, block_layout.texels_per_row (), block_layout.texels_per_column ());
    glMatrixMode (GL_PROJECTION);
    glLoadIdentity ();
    glOrtho (br.a.time, br.b.time, br.a.scale, br.b.scale, -10,10);
    glMatrixMode (GL_MODELVIEW);
    glLoadIdentity ();

    GlException_CHECK_ERROR ();

    // Disable unwanted capabilities when resampling a texture
    glDisable (GL_DEPTH_TEST);
    glDisable (GL_BLEND);
    glDisable (GL_CULL_FACE);
}


void BlockFbo::
        end ()
{
    fbo->unbindFrameBuffer ();
}


ChunkToBlockDegenerateTexture::Shader::Shader(GLuint program)
    :
      program(program),
      normalization_location_(-1),
      amplitude_axis_location_(-1),
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
    GlException_SAFE_CALL( glUniform1i(mytex, 0) ); // mytex corresponds to GL_TEXTURE0
    GlException_SAFE_CALL( glUseProgram(0) );
}


ChunkToBlockDegenerateTexture::Shader::~Shader()
{
    if (program)
        glDeleteProgram(program);
}


void ChunkToBlockDegenerateTexture::Shader::Shader::
        setParams(int data_width, int data_height, int tex_width, int tex_height,
               float normalization_factor, int amplitude_axis)
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
    GlException_SAFE_CALL( glUseProgram(0) );
}


ChunkToBlockDegenerateTexture::Shaders::Shaders():
    chunktoblock_shader_(ShaderResource::loadGLSLProgram("", ":/shaders/chunktoblock.frag")),
    chunktoblock_maxwidth_shader_(ShaderResource::loadGLSLProgram("", ":/shaders/chunktoblock_maxwidth.frag")),
    chunktoblock_maxheight_shader_(ShaderResource::loadGLSLProgram("", ":/shaders/chunktoblock_maxheight.frag"))
{
}


ChunkToBlockDegenerateTexture::ShaderTexture::ShaderTexture(Shaders &shaders)
    :
      shaders_(shaders)
{
}


void ChunkToBlockDegenerateTexture::ShaderTexture::
        prepareShader (int data_width, int data_height, float* p)
{
    prepareShader(data_width, data_height, 0, p);
}


void ChunkToBlockDegenerateTexture::ShaderTexture::
        prepareShader (int data_width, int data_height, unsigned chunk_pbo_)
{
    prepareShader(data_width, data_height, chunk_pbo_, 0);
}


const GlTexture& ChunkToBlockDegenerateTexture::ShaderTexture::
        getTexture ()
{
    EXCEPTION_ASSERT(chunk_texture_);

    return *chunk_texture_;
}

unsigned ChunkToBlockDegenerateTexture::ShaderTexture::
        getProgram (float normalization_factor, int amplitude_axis)
{
    shader_->setParams (data_width, data_height, tex_width, tex_height, normalization_factor, amplitude_axis);
    return shader_->program;
}


void ChunkToBlockDegenerateTexture::ShaderTexture::
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


ChunkToBlockDegenerateTexture::DrawableChunk::DrawableChunk(
        Tfr::pChunk chunk,
        const Parameters& params,
        BlockFbos& block_fbos,
        Shaders& shaders
        )
    :
        chunk_(chunk->transform_data),
        params_(params),
        block_fbos_(block_fbos),
        shader_(shaders),
        mapped_chunk_data_(0),
        vbo_(0),
        chunk_pbo_(0),
        sync_(0)
{
    EXCEPTION_ASSERT(chunk);

    chunk_scale = chunk->freqAxis;
    nScales = chunk->nScales ();
    nSamples = chunk->nSamples ();
    nValidSamples = chunk->n_valid_samples;
    transpose = chunk->order == Tfr::Chunk::Order_column_major;
    data_width  = transpose ? nScales : nSamples;
    data_height = transpose ? nSamples : nScales;

    Signal::Interval inInterval = chunk->getCoveredInterval();
    INFO TaskTimer tt(boost::format("ChunkToBlockDegenerateTexture(%s) %u x %u <> %d")
                      % inInterval % nSamples % nScales % gl_max_texture_size());

    a_t = inInterval.first / chunk->original_sample_rate;
    b_t = inInterval.last / chunk->original_sample_rate;

    if (transpose)
      {
         u0 = 0.f / nSamples; // The sample at a_t
         u1 = (nSamples-1.0f) / nSamples; // The sample at b_t
      }
    else
      {
        u0 = chunk->first_valid_sample / (float)nSamples;
        u1 = (chunk->first_valid_sample+chunk->n_valid_samples-1.0f) / (float)nSamples;
      }

    setupPbo ();
    setupVbo ();

//    sync_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);

    EXCEPTION_ASSERT (chunk_pbo_);
}


ChunkToBlockDegenerateTexture::DrawableChunk::
        ~DrawableChunk()
{
    if (mapped_chunk_data_)
    {
        TaskInfo("Waiting for data_transfer before releasing gl resources");
        data_transfer.wait ();
    }

    if (vbo_)
        glDeleteBuffers (1, &vbo_);

    if (chunk_pbo_)
        glDeleteBuffers (1, &chunk_pbo_);

//    if (sync_)
//        glDeleteSync((GLsync)(void*)sync_);
}


std::packaged_task<void()> ChunkToBlockDegenerateTexture::DrawableChunk::
        transferData(float *p)
{
    // http://www.seas.upenn.edu/~pcozzi/OpenGLInsights/OpenGLInsights-AsynchronousBufferTransfers.pdf

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_);
    mapped_chunk_data_ = (float*)glMapBuffer (GL_PIXEL_UNPACK_BUFFER, GL_WRITE_ONLY);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    int n = data_width*data_height;

    float *c = mapped_chunk_data_;
    auto t = std::packaged_task<void()>([c, p, n](){
//        Timer t;
        memcpy(c, p, n*sizeof(float));
//        TaskInfo("memcpy %s with %s/s", DataStorageVoid::getMemorySizeText(n*sizeof(float)).c_str (), DataStorageVoid::getMemorySizeText(n*sizeof(float) / t.elapsed ()).c_str ());
    });

    data_transfer = t.get_future();
    return t;
}


void ChunkToBlockDegenerateTexture::DrawableChunk::
        setupPbo ()
{
    int n = data_width*data_height;

    glGenBuffers (1, &chunk_pbo_); // Generate 1 buffer
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, sizeof(float)*n, 0, GL_STATIC_DRAW);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);

    GlException_CHECK_ERROR();
}


void ChunkToBlockDegenerateTexture::DrawableChunk::
        setupVbo()
{
    INFO TaskTimer tt(boost::format("ChunkToBlockDegenerateTexture::prepVbo %u x %u -> %s")
                      % nSamples % nScales % params_.block_layout);

    float a_t = this->a_t; // The first sample should be centered on a_t
    float b_t = this->b_t; // The last sample should be centered on b_t
    float u0 = this->u0; // Normalized index for sample at a_t (not the texture coord to use right away)
    float u1 = this->u1; // Normalized index for sample at b_t (not the texture coord to use right away)
    unsigned Y = nScales;
    bool transpose = this->transpose;
    const Tfr::FreqAxis chunk_scale = this->chunk_scale;
    const Tfr::FreqAxis display_scale = this->params_.display_scale;

//    float min_hz = chunk_scale.getFrequency (0u);
//    float max_hz = chunk_scale.getFrequency (Y);

//    Region chunk_region = Region(Position(),Position());
//    chunk_region.a.time = a_t;
//    chunk_region.b.time = b_t;
//    chunk_region.a.scale = display_scale.getFrequencyScalar (min_hz);
//    chunk_region.b.scale = display_scale.getFrequencyScalar (max_hz);
//    INFO TaskTimer tt(boost::format("Creating VBO for %s") % chunk_region);

    // Build VBO contents
    std::vector<float> vertices((Y*2)*sizeof(vertex_format)/sizeof(float));
    int i=0;

    // Juggle texture coordinates so that border texels are centered on the border
    float iY = 1.f / Y;
//    float dt = b_t - a_t;
//    a_t -= 0.5 * dt / (nSamples-1);
//    b_t += 0.5 * dt / (nSamples-1);
//    float ky = 1.f + 1.f/(Y-1.f);
//    float oy = 0.5f;

    for (unsigned y=0; y<Y; y++)
      {
//        float hz = chunk_scale.getFrequency (y * ky - oy);
        float hz = chunk_scale.getFrequency (y);
        if (hz < display_scale.min_hz)
            hz = display_scale.min_hz/2;
        float s = display_scale.getFrequencyScalar(hz);
        float v = (y + 0.0)*iY;

        vertices[i++] = a_t;
        vertices[i++] = s;
        vertices[i++] = transpose ? v : u0; // Normalized index
        vertices[i++] = transpose ? u0 : v;
        vertices[i++] = b_t;
        vertices[i++] = s;
        vertices[i++] = transpose ? v : u1;
        vertices[i++] = transpose ? u1 : v;
      }

    if (vbo_)
      {
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, vbo_) );
        GlException_SAFE_CALL( glBufferSubData (GL_ARRAY_BUFFER, 0, sizeof(float)*vertices.size (), &vertices[0]) );
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, 0) );
      }
    else
      {
        GlException_SAFE_CALL( glGenBuffers (1, &vbo_) ); // Generate 1 buffer
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, vbo_) );
        GlException_SAFE_CALL( glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertices.size (), &vertices[0], GL_STREAM_DRAW) );
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, 0) );
      }
}


bool ChunkToBlockDegenerateTexture::DrawableChunk::
        has_chunk() const
{
    return chunk_pbo_;
}


bool ChunkToBlockDegenerateTexture::DrawableChunk::
        ready() const
{
    return data_transfer.valid ();
//    GLint result = GL_UNSIGNALED;
//    glGetSynciv((GLsync)(void*)sync_, GL_SYNC_STATUS, sizeof(GLint), NULL, &result);
//    return result == GL_SIGNALED;
}


void ChunkToBlockDegenerateTexture::DrawableChunk::
        prepareShader ()
{
    if (mapped_chunk_data_)
    {
        if (!data_transfer.valid ())
        {
            int n = data_width*data_height;
            TaskTimer tt("data_transfer.wait () %d", n);
            data_transfer.wait ();
        }
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, chunk_pbo_);
        glUnmapBuffer (GL_PIXEL_UNPACK_BUFFER);
        glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
        mapped_chunk_data_ = 0;

        shader_.prepareShader (data_width, data_height, chunk_pbo_);
    }
}


void ChunkToBlockDegenerateTexture::DrawableChunk::
        draw ()
{
    EXCEPTION_ASSERT (chunk_pbo_);

    prepareShader ();

    GlException_CHECK_ERROR ();

    INFO TaskTimer tt("ChunkToBlockDegenerateTexture::mergeChunk");

    // Setup drawing with VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexPointer(2, GL_FLOAT, sizeof(vertex_format), 0);
    glTexCoordPointer(2, GL_FLOAT, sizeof(vertex_format), (float*)0 + 2);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    GlException_CHECK_ERROR();

    // Setup the shader
    glUseProgram(shader_.getProgram (params_.normalization_factor, (int)params_.amplitude_axis));
    const GlTexture& chunk_texture = shader_.getTexture ();

    // Draw from chunk texture onto block texture with framebuffer rendering
    {
        chunk_texture.bindTexture2D ();

        glDrawArrays(GL_TRIANGLE_STRIP, 0, nScales*2);

        chunk_texture.unbindTexture2D ();
    }

    // Finish with shader
    glUseProgram(0);

    // Finish drawing with VBO
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GlException_CHECK_ERROR();
}


ChunkToBlockDegenerateTexture::
        ChunkToBlockDegenerateTexture()
    :
      params_ {AmplitudeAxis(), Tfr::FreqAxis(), BlockLayout(2,2,1), 1}
{
}


ChunkToBlockDegenerateTexture::DrawableChunk ChunkToBlockDegenerateTexture::
        prepareChunk(Tfr::pChunk chunk)
{
    EXCEPTION_ASSERT( chunk );
    EXCEPTION_ASSERT_NOTEQUALS( params_.display_scale.axis_scale, Tfr::AxisScale_Unknown);

    return DrawableChunk( chunk, params_, block_fbos_, shaders_ );
}


ChunkToBlockDegenerateTexture::
        ~ChunkToBlockDegenerateTexture()
{
}


void ChunkToBlockDegenerateTexture::
        setParams(AmplitudeAxis amplitude_axis, Tfr::FreqAxis display_scale, BlockLayout block_layout, float normalization_factor)
{
    Parameters new_params {amplitude_axis, display_scale, block_layout, normalization_factor};

    if (params_ == new_params)
        return;

    params_ = new_params;
}


void ChunkToBlockDegenerateTexture::
        prepareBlock (pBlock block)
{
    if (block_fbos_.count (block))
        return;

    EXCEPTION_ASSERT (block->glblock);

    std::shared_ptr<BlockFbo> fbo(new BlockFbo(block));
    block_fbos_.insert (std::pair<pBlock, std::shared_ptr<BlockFbo>>(block, fbo));
}


void ChunkToBlockDegenerateTexture::
        finishBlocks ()
{
    block_fbos_.clear ();
}

} // namespace Heightmap



namespace Heightmap {

void ChunkToBlockDegenerateTexture::
        test()
{
    // It should merge the contents of a chunk directly onto the texture of a block.
    {

    }
}

} // namespace Heightmap


/*    float xstep = gl_TexCoord[0].p;
    if (xstep < 0.5)
    {
        // Translate normalized index to data index (integers)
        vec2 uv = floor(gl_TexCoord[0].st * data_size);
        // Compute neighbouring indices as well, and their distance
        vec2 f = gl_TexCoord[0].st * data_size - uv;
        vec4 u = vec4(uv.x, uv.x+1.0, uv.x, uv.x+1.0);
        vec4 v = vec4(uv.y, uv.y, uv.y+1.0, uv.y+1.0);

        // Compute linear index common for data and degenerate texture
        // With IEEE-754 single floats 'i' is an exact integer up to 16 million.
        // But that's only guaranteed in GLSL 1.30 and above.
        vec4 i = u + v*data_size.x;

        // Compute degenerate texel index (integers)
        v = floor(i / tex_size.x);
        u = mod(i, tex_size.x);

        // Compute texture position that will make a nearest lookup
        // find the right texel in degenerate texture
        u = (u + 0.5) / tex_size.x;
        v = (v + 0.5) / tex_size.y;

        vec4 r = vec4(
                    texture2D(mytex, vec2(u.s, v.s)).r,
                    texture2D(mytex, vec2(u.t, v.t)).r,
                    texture2D(mytex, vec2(u.p, v.p)).r,
                    texture2D(mytex, vec2(u.q, v.q)).r );

        r.xy = mix(r.xz, r.yw, f.x);
        a = mix(r.x, r.y, f.y);
    }*/
/*    else
    {
        // Translate normalized index to data index (integers)
        vec2 xrange = (gl_TexCoord[0].xx + gl_TexCoord[0].p*vec2(-1.0,1.0)) * data_size.x;
        xrange = floor(xrange + 0.5);
        float iy = floor(gl_TexCoord[0].y * data_size.y);
        float fy = gl_TexCoord[0].y * data_size.y - iy;
        vec2 y = vec2(iy, iy+1);

        // Assume the primary resolution has the highest resolution and only implement max along that.
        // Interpolate the other.

        vec2 av = vec2(0.0, 0.0);
        for (float x=xrange.x; x<=xrange.y; ++x)
        {
            // Compute linear index common for data and degenerate texture
            // With IEEE-754 single floats 'i' is an exact integer up to 16 million.
            // But that's only guaranteed in GLSL 1.30 and above.
            vec2 i = x + y*data_size.x;

            // Compute degenerate texel index (integers)
            vec2 v = floor(i / tex_size.x);
            vec2 u = mod(i, tex_size.x);

            // Compute texture position that will make a nearest lookup
            // find the right texel in degenerate texture
            u = (u + 0.5) / tex_size.x;
            v = (v + 0.5) / tex_size.y;

            vec2 t = vec2(texture2D(mytex, vec2(u.x, v.x)).r,
                          texture2D(mytex, vec2(u.y, v.y)).r);
            av = max(av, t);
        }

        a = mix(av.x, av.y, fy);
    }*/

    /*{
        // Translate normalized index to data index (integers)
        vec2 uv1 = (gl_TexCoord[0].st - vec2(gl_TexCoord[0].p,0)) * data_size;
        vec2 uv2 = (gl_TexCoord[0].st + vec2(gl_TexCoord[0].p,0)) * data_size;

        // Assume the primary resolution has the highest resolution and only implement max along that.
        // Interpolate the other.

        // Compute neighbouring indices as well, and their distance
        float a = 0.0;
        vec2 uv;
        for (uv.x=floor(uv1.x); uv.x<=ceil(uv2.x); ++uv.x) {
            for (uv.y=floor(uv1.y); uv.y<=ceil(uv2.y); ++uv.y)
            {
                // Compute linear index common for data and degenerate texture
                // With IEEE-754 single floats 'i' is an exact integer up to 16 million.
                // But that's only guaranteed in GLSL 1.30 and above.
                float i = uv.x + uv.y*data_size.x;

                // Compute degenerate texel index (integers)
                float v = floor(i / tex_size.x);
                float u = mod(i, tex_size.x);

                // Compute texture position that will make a nearest lookup
                // find the right texel in degenerate texture
                u = (u + 0.5) / tex_size.x;
                v = (v + 0.5) / tex_size.y;

                a = max(a, texture2D(mytex, vec2(u, v)).r);
            }
        }
    }*/
