#include "chunktoblockdegeneratetexture.h"
#include "glblock.h"
#include "tfr/chunk.h"

#include "glframebuffer.h"
#include "TaskTimer.h"
#include "gl.h"
#include "GlException.h"
#include "shaderresource.h"
#include "glPushContext.h"
#include "gltextureread.h"
#include "neat_math.h"
#include "factor.h"

//#define INFO
#define INFO if(0)

namespace Heightmap {

struct vertex_format {
    float x, y, u, v;
};


ChunkToBlockDegenerateTexture::
        ChunkToBlockDegenerateTexture(Tfr::pChunk chunk)
    :
      vbo_(0)
{
    int tex_height, tex_width, data_width, data_height;

    nScales = chunk->nScales ();
    nSamples = chunk->nSamples ();
    nValidSamples = chunk->n_valid_samples;
    transpose = chunk->order == Tfr::Chunk::Order_column_major;
    data_width  = transpose ? nScales : nSamples;
    data_height = transpose ? nSamples : nScales;
    Tfr::ChunkElement *p = chunk->transform_data->getCpuMemory ();
    // Compute the norm of the chunk prior to resampling and interpolating
    int n = chunk->transform_data->numberOfElements ();
    for (int i = 0; i<n; ++i)
        p[i] = Tfr::ChunkElement( norm(p[i]), 0.f );

    Signal::Interval inInterval = chunk->getCoveredInterval();
    INFO TaskTimer tt(boost::format("ChunkToBlockDegenerateTexture. Creating texture for chunk %s with nSamples=%u, nScales=%u")
                      % inInterval % nSamples % nScales);

    int gl_max_texture_size = 0;
    glGetIntegerv (GL_MAX_TEXTURE_SIZE, &gl_max_texture_size);
    gl_max_texture_size--; // This seems to be the largest allowed texture in practice...
    if (data_width <= gl_max_texture_size && data_height <= gl_max_texture_size)
      {
        // No need for degenerate (could still use degeneracy to make the width
        // (and height) a power of 2 if it would help performance)
        shader_ = ShaderResource::loadGLSLProgram("", ":/shaders/chunktoblock.frag");
        tex_width = data_width;
        tex_height = data_height;

        chunk_texture_.reset (new GlTexture( tex_width, tex_height, GL_RG, GL_RED, GL_FLOAT, p));
      }
    else
      {
        int n = data_width*data_height;
        EXCEPTION_ASSERT_EQUALS((unsigned)n, chunk->transform_data->numberOfElements ());

        if (data_width > gl_max_texture_size)
          {
            // Degeneracy. Make the size a multiple of data_height.
            shader_ = ShaderResource::loadGLSLProgram("", ":/shaders/chunktoblock_maxwidth.frag");

            tex_width = gl_max_texture_size;
            int s = int_div_ceil (data_width, (tex_width-1));
            tex_height = data_height * s;

            // Out of open gl memory if this happens...
            EXCEPTION_ASSERT_LESS_OR_EQUAL(tex_height, gl_max_texture_size);

            chunk_texture_.reset (new GlTexture( tex_width, tex_height, GL_RG, GL_RED, GL_FLOAT, 0));

            GlTexture::ScopeBinding texObjBinding = chunk_texture_->getScopeBinding();
            for (int i=0; i<s; ++i)
              {
                for (int h=0; h<data_height; ++h)
                  {
                    int w = std::min(tex_width, data_width - (tex_width-1)*i);
                    int y = h + i*data_height;
                    Tfr::ChunkElement* q = &p[h*data_width + i*(tex_width-1)];
                    glTexSubImage2D (GL_TEXTURE_2D, 0, 0, y, w, 1, GL_RG, GL_FLOAT, q);
                  }
              }

            GlException_CHECK_ERROR();
          }
        else
          {
            shader_ = ShaderResource::loadGLSLProgram("", ":/shaders/chunktoblock_maxheight.frag");

            tex_height = gl_max_texture_size;
            int s = int_div_ceil (data_height, (tex_height-1));
            tex_width = data_width * s;

            // Out of open gl memory if this happens...
            EXCEPTION_ASSERT_LESS_OR_EQUAL(tex_width, gl_max_texture_size);

            chunk_texture_.reset (new GlTexture( tex_width, tex_height, GL_RG, GL_RED, GL_FLOAT, 0));

            GlTexture::ScopeBinding texObjBinding = chunk_texture_->getScopeBinding();
            for (int i=0; i<s; ++i)
              {
                int h = std::min(tex_height, data_height - (tex_height-1)*i);
                int w = data_width;
                int x = i*data_width;
                Tfr::ChunkElement* q = &p[data_width*i*(tex_height-1)];
                glTexSubImage2D (GL_TEXTURE_2D, 0, x, 0, w, h, GL_RG, GL_FLOAT, q);
              }

            GlException_CHECK_ERROR();
          }
      }


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

    chunk_scale = chunk->freqAxis;

    GlException_SAFE_CALL( glUseProgram(shader_) );
    int mytex = glGetUniformLocation(shader_, "mytex");
    GlException_CHECK_ERROR();
    int data_size_loc = glGetUniformLocation(shader_, "data_size");
    GlException_CHECK_ERROR();
    int tex_size_loc = glGetUniformLocation(shader_, "tex_size");
    GlException_CHECK_ERROR();
    normalization_location_ = glGetUniformLocation(shader_, "normalization");
    GlException_CHECK_ERROR();
    amplitude_axis_location_ = glGetUniformLocation(shader_, "amplitude_axis");
    GlException_SAFE_CALL( glUniform1i(mytex, 0) ); // mytex corresponds to GL_TEXTURE0
    if ( 0 <= data_size_loc)
        GlException_SAFE_CALL( glUniform2f(data_size_loc, data_width, data_height) );
    if ( 0 <= tex_size_loc)
        GlException_SAFE_CALL( glUniform2f(tex_size_loc, tex_width, tex_height) );
    GlException_SAFE_CALL( glUseProgram(0) );
}


void ChunkToBlockDegenerateTexture::
        prepVbo(Tfr::FreqAxis display_scale, BlockLayout bl)
{
    if (this->display_scale == display_scale)
        return;
    this->display_scale = display_scale;

    float a_t = this->a_t; // The first sample should be centered on a_t
    float b_t = this->b_t; // The last sample should be centered on b_t
    float u0 = this->u0; // Normalized index for sample at a_t (not the texture coord to use right away)
    float u1 = this->u1; // Normalized index for sample at b_t (not the texture coord to use right away)
    unsigned Y = nScales;
    bool transpose = this->transpose;
    const Tfr::FreqAxis& chunk_scale = this->chunk_scale;

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


ChunkToBlockDegenerateTexture::
        ~ChunkToBlockDegenerateTexture()
{
    glDeleteBuffers (1, &vbo_);

    glUseProgram(0);
    glDeleteProgram(shader_);
}


void ChunkToBlockDegenerateTexture::
        mergeChunk( const pBlock pblock )
{
    boost::shared_ptr<GlBlock> glblock = pblock->glblock;
    if (!glblock)
      {
        // This block was garbage collected after the list of matching blocks were created.
        return;
      }

    Block& block = *pblock;
    Region br = block.getRegion ();

    INFO TaskTimer tt(boost::format("ChunkToBlockDegenerateTexture::mergeChunk %s") % br);

    GlException_CHECK_ERROR();

    // Juggle texture coordinates so that border texels are centered on the border
    const BlockLayout bl = block.block_layout ();
    float dt = br.time (), ds = br.scale ();
    br.a.time -= 0.5*dt / bl.texels_per_row ();
    br.b.time += 0.5*dt / bl.texels_per_row ();
    br.a.scale -= 0.5*ds / bl.texels_per_column ();
    br.b.scale += 0.5*ds / bl.texels_per_column ();

    // Setup the VBO, need to know the current display scale, which is defined by the block.
    VisualizationParams::ConstPtr vp = block.visualization_params ();
    prepVbo(vp->display_scale (), bl);

    // Disable unwanted capabilities when resampling a texture
    glPushAttribContext pa(GL_ENABLE_BIT);
    glDisable (GL_DEPTH_TEST);
    glDisable (GL_BLEND);
    glDisable (GL_CULL_FACE);

    // Setup matrices, while preserving the old ones
    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    GlException_SAFE_CALL( glViewport(0, 0, bl.texels_per_row (), bl.texels_per_column ()) );
    glPushMatrixContext mpc( GL_PROJECTION );
    glLoadIdentity();
    glOrtho(br.a.time, br.b.time, br.a.scale, br.b.scale, -10,10);
    glPushMatrixContext mc( GL_MODELVIEW );
    glLoadIdentity();

    // Setup drawing with VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glVertexPointer(2, GL_FLOAT, sizeof(vertex_format), 0);
    glTexCoordPointer(2, GL_FLOAT, sizeof(vertex_format), (float*)0 + 2);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    // Setup the shader
    glUseProgram(shader_);
    glUniform1f(normalization_location_, normalization_factor);
    glUniform1i(amplitude_axis_location_, (int)vp->amplitude_axis());

    // Draw from chunk texture onto block texture
    {
        // Setup framebuffer rendering
        GlTexture::Ptr t = glblock->glTexture ();
        GlFrameBuffer fbo(t->getOpenGlTextureId ());
        GlFrameBuffer::ScopeBinding sb = fbo.getScopeBinding ();

        GlTexture::ScopeBinding texObjBinding = chunk_texture_->getScopeBinding();
        GlException_SAFE_CALL( glDrawArrays(GL_TRIANGLE_STRIP, 0, nScales*2) );

        // Copy to vertex texture
        t = glblock->glVertTexture ();
        glBindTexture(GL_TEXTURE_2D, t->getOpenGlTextureId ());
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0,0, 0,0, t->getWidth (), t->getHeight ());
        glBindTexture(GL_TEXTURE_2D, 0);

    }

    // Finish with shader
    glUseProgram(0);

    // Finish drawing with WBO
    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    // Restore old matrices
    GlException_SAFE_CALL( glViewport(viewport[0], viewport[1], viewport[2], viewport[3] ) );

    const_cast<Block*>(&block)->discard_new_block_data ();

    GlException_CHECK_ERROR();
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
