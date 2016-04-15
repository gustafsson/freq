#include "texture2fbo.h"
#include "tfr/chunk.h"
#include "GlException.h"
#include "glstate.h"
#include "tasktimer.h"
#include "glgroupmarker.h"
#include "log.h"

//#define INFO
#define INFO if(0)

namespace Heightmap {
namespace Update {
namespace OpenGL {

struct vertex_format {
    float x, y, u, v;
};

Texture2Fbo::Params::
        Params(Tfr::pChunk chunk,
               Heightmap::Region region,
               Heightmap::FreqAxis display_scale,
               Heightmap::BlockLayout block_layout)
    :
      block_layout(block_layout),
      region(region),
      display_scale(display_scale),
      chunk_scale(chunk->freqAxis)
{
    EXCEPTION_ASSERT(chunk);
    QOpenGLFunctions::initializeOpenGLFunctions ();

    nScales = chunk->nScales ();
    nSamples = chunk->nSamples ();
    nValidSamples = chunk->n_valid_samples;
    transpose = chunk->order == Tfr::Chunk::Order_column_major;
    data_width  = transpose ? nScales : nSamples;
    data_height = transpose ? nSamples : nScales;

    Signal::Interval inInterval = chunk->getCoveredInterval();
    INFO TaskTimer tt(boost::format("ChunkToBlockDegenerateTexture(%s) %u x %u")
                      % inInterval % nSamples % nScales);

    a_t = inInterval.first / chunk->original_sample_rate;
    b_t = inInterval.last / chunk->original_sample_rate;

    if (transpose)
      {
         u0 = 0.f / nSamples; // The sample at a_t
         u1 = (nValidSamples-1.0f) / nSamples; // The sample at b_t
      }
    else
      {
        u0 = chunk->first_valid_sample / (float)nSamples;
        u1 = (chunk->first_valid_sample+chunk->n_valid_samples-1.0f) / (float)nSamples;
      }
}


int Texture2Fbo::Params::
        createVbo(int& out_num_elements) const
{
    unsigned vbo = 0;

    INFO TaskTimer tt(boost::format("Texture2Fbo::setupVbo %u x %u -> %s")
                      % nSamples % nScales % block_layout);

    float a_t = this->a_t; // The first sample should be centered on a_t
    float b_t = this->b_t; // The last sample should be centered on b_t
    float u0 = this->u0; // Normalized index for sample at a_t (not the texture coord to use right away)
    float u1 = this->u1; // Normalized index for sample at b_t (not the texture coord to use right away)
    int Y = block_layout.texels_per_column ();
    bool transpose = this->transpose;
    const Tfr::FreqAxis chunk_scale = this->chunk_scale;
    const Heightmap::FreqAxis display_scale = this->display_scale;

    // Build VBO contents
    if (vbo)
      {
        GlException_SAFE_CALL( GlState::glBindBuffer(GL_ARRAY_BUFFER, vbo) );
      }
    else
      {
        GlException_SAFE_CALL( glGenBuffers (1, &vbo) ); // Generate 1 buffer
        GlException_SAFE_CALL( GlState::glBindBuffer(GL_ARRAY_BUFFER, vbo) );
        GlException_SAFE_CALL( glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_format)*Y*2, 0, GL_STATIC_DRAW) );
      }

#if !defined(LEGACY_OPENGL) || defined(GL_ES_VERSION_3_0)
    vertex_format* vertices = (vertex_format*)glMapBufferRange(GL_ARRAY_BUFFER, 0, Y*2*sizeof(vertex_format), GL_MAP_INVALIDATE_RANGE_BIT | GL_MAP_WRITE_BIT);
#else
    vertex_format* vertices = (vertex_format*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
#endif

    float iY = (region.b.scale - region.a.scale) / (Y-1.f);
    float iV = 1.f/(chunk_scale.max_frequency_scalar+1);

    float hz0 = chunk_scale.getFrequency (0.f);
    float hz1 = chunk_scale.getFrequency (chunk_scale.max_frequency_scalar);
    float s0 = display_scale.getFrequencyScalarNotClamped (hz0);
    float s1 = display_scale.getFrequencyScalarNotClamped (hz1);

    float prev_s = -1;
    out_num_elements = 0;

    for (int y=0; y<Y; y++)
      {
        float s = region.a.scale + y*iY;
        float hz = display_scale.getFrequency (s);
        float v = chunk_scale.getFrequencyScalarNotClamped (hz);

        if (v<0)
        {
            v = 0;
            s = s0;
        }
        if (v>chunk_scale.max_frequency_scalar)
        {
            v = chunk_scale.max_frequency_scalar;
            s = s1;
        }

        // Normalized index. Texel lookup in chunktoblock*.frag
        v = v*iV;

        if (prev_s != s)
        {
            prev_s = s;
            *vertices++ = vertex_format{ a_t, s, transpose ? v : u0, transpose ? u0 : v};
            *vertices++ = vertex_format{ b_t, s, transpose ? v : u1, transpose ? u1 : v};
            out_num_elements += 2;
        }
      }

    glUnmapBuffer(GL_ARRAY_BUFFER);
    GlException_SAFE_CALL( GlState::glBindBuffer(GL_ARRAY_BUFFER, 0) );

    return vbo;
}


Texture2Fbo::Texture2Fbo(
        const Params& p, float normalization_factor
        )
    :
      normalization_factor_(normalization_factor)
{
    vbo_ = p.createVbo (num_elements_);

    //    sync_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    EXCEPTION_ASSERT (vbo_);
}


Texture2Fbo::~Texture2Fbo()
{
    if (!QOpenGLContext::currentContext ()) {
        Log ("%s: destruction without gl context leaks vbo %d") % __FILE__ % unsigned(vbo_);
        return;
    }

    if (vbo_)
        GlState::glDeleteBuffers (1, &vbo_);
}


void Texture2Fbo::
        draw (int vertex_attrib, int tex_attrib) const
{
    GlException_CHECK_ERROR ();

    INFO TaskTimer tt("Texture2Fbo::draw");
    GlGroupMarker gpm("Texture2Fbo::draw");

    // Setup drawing with VBO
    GlState::glBindBuffer(GL_ARRAY_BUFFER, vbo_);

    GlState::glEnableVertexAttribArray (vertex_attrib);
    GlState::glEnableVertexAttribArray (tex_attrib);
    glVertexAttribPointer (vertex_attrib, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), 0);
    glVertexAttribPointer (tex_attrib, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), (float*)0 + 2);

    GlException_CHECK_ERROR();

    // Draw with vbo onto block texture with framebuffer rendering
    GlState::glDrawArrays(GL_TRIANGLE_STRIP, 0, num_elements_);

    // Finish drawing with VBO
    GlState::glDisableVertexAttribArray (vertex_attrib);
    GlState::glDisableVertexAttribArray (tex_attrib);
    GlState::glBindBuffer(GL_ARRAY_BUFFER, 0);

    GlException_CHECK_ERROR();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
