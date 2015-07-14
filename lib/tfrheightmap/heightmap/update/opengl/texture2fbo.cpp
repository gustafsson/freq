#include "texture2fbo.h"
#include "tfr/chunk.h"
#include "GlException.h"
#include "gl.h"
#include "tasktimer.h"

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
               Heightmap::FreqAxis display_scale,
               Heightmap::BlockLayout block_layout)
    :
      block_layout(block_layout)
{
    // to allow memcmp, the type is not densely packed
    memset(this, 0, sizeof(Params));

    EXCEPTION_ASSERT(chunk);
    this->block_layout = block_layout;
    this->display_scale = display_scale;

    chunk_scale = chunk->freqAxis;
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
        createVbo() const
{
    unsigned vbo = 0;

    INFO TaskTimer tt(boost::format("Texture2Fbo::setupVbo %u x %u -> %s")
                      % nSamples % nScales % block_layout);

    float a_t = this->a_t; // The first sample should be centered on a_t
    float b_t = this->b_t; // The last sample should be centered on b_t
    float u0 = this->u0; // Normalized index for sample at a_t (not the texture coord to use right away)
    float u1 = this->u1; // Normalized index for sample at b_t (not the texture coord to use right away)
    unsigned Y = nScales;
    bool transpose = this->transpose;
    const Tfr::FreqAxis chunk_scale = this->chunk_scale;
    const Heightmap::FreqAxis display_scale = this->display_scale;

//    float min_hz = chunk_scale.getFrequency (0u);
//    float max_hz = chunk_scale.getFrequency (Y);

//    Region chunk_region = Region(Position(),Position());
//    chunk_region.a.time = a_t;
//    chunk_region.b.time = b_t;
//    chunk_region.a.scale = display_scale.getFrequencyScalar (min_hz);
//    chunk_region.b.scale = display_scale.getFrequencyScalar (max_hz);
//    INFO TaskTimer tt(boost::format("Creating VBO for %s") % chunk_region);

    // Build VBO contents
    std::vector<vertex_format> vertices(Y*2);
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

        vertices[i].x = a_t;
        vertices[i].y = s;
        vertices[i].u = transpose ? v : u0; // Normalized index
        vertices[i].v = transpose ? u0 : v;
        i++;
        vertices[i].x = b_t;
        vertices[i].y = s;
        vertices[i].u = transpose ? v : u1;
        vertices[i].v = transpose ? u1 : v;
        i++;
      }

    if (vbo)
      {
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, vbo) );
        GlException_SAFE_CALL( glBufferSubData (GL_ARRAY_BUFFER, 0, sizeof(vertex_format)*vertices.size (), &vertices[0]) );
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, 0) );
      }
    else
      {
        GlException_SAFE_CALL( glGenBuffers (1, &vbo) ); // Generate 1 buffer
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, vbo) );
        GlException_SAFE_CALL( glBufferData(GL_ARRAY_BUFFER, sizeof(vertex_format)*vertices.size (), &vertices[0], GL_STATIC_DRAW) );
        GlException_SAFE_CALL( glBindBuffer(GL_ARRAY_BUFFER, 0) );
      }

    return vbo;
}


int Texture2Fbo::Params::
        numVboElements() const
{
    return 2*nScales;
}


Texture2Fbo::Texture2Fbo(
        const Params& p, float normalization_factor
        )
    :
      normalization_factor_(normalization_factor)
{
    vbo_ = p.createVbo ();
    num_elements_ = p.numVboElements();

    //    sync_ = glFenceSync(GL_SYNC_GPU_COMMANDS_COMPLETE, 0);
    EXCEPTION_ASSERT (vbo_);
}


Texture2Fbo::~Texture2Fbo()
{
    if (vbo_)
        glDeleteBuffers (1, &vbo_);
}


void Texture2Fbo::
        draw (int vertex_attrib, int tex_attrib) const
{
    GlException_CHECK_ERROR ();

    INFO TaskTimer tt("ChunkToBlockDegenerateTexture::mergeChunk");

    // Setup drawing with VBO
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);

    glEnableVertexAttribArray (vertex_attrib);
    glEnableVertexAttribArray (tex_attrib);
    glVertexAttribPointer (vertex_attrib, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), 0);
    glVertexAttribPointer (tex_attrib, 2, GL_FLOAT, GL_TRUE, sizeof(vertex_format), (float*)0 + 2);

    GlException_CHECK_ERROR();

    // Draw with vbo onto block texture with framebuffer rendering
    glDrawArrays(GL_TRIANGLE_STRIP, 0, num_elements_);

    // Finish drawing with VBO
    glDisableVertexAttribArray (vertex_attrib);
    glDisableVertexAttribArray (tex_attrib);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    GlException_CHECK_ERROR();
}


} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
