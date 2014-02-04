#include "chunktoblocktexture.h"
#include "glblock.h"
#include "tfr/chunk.h"

#include "glframebuffer.h"
#include "TaskTimer.h"
#include "gl.h"
#include "GlException.h"
#include "shaderresource.h"
#include "glPushContext.h"
#include "gltextureread.h"
#include "datastoragestring.h"

//#define PRINT_TEXTURES
#define PRINT_TEXTURES if(0)

//#define INFO
#define INFO if(0)

namespace Heightmap {

ChunkToBlockTexture::
        ChunkToBlockTexture()
{
    glGenBuffers (1, &vbo_); // Generate 1 buffer

    shader_ = ShaderResource::loadGLSLProgram("", ":/shaders/complex_magnitude.frag");
    glUseProgram(shader_);
    int loc = glGetUniformLocation(shader_, "mytex");
    normalization_location_ = glGetUniformLocation(shader_, "normalization");
    glUniform1i(loc, 0); // mytex corresponds to GL_TEXTURE0
    glUseProgram(0);
}


ChunkToBlockTexture::
        ~ChunkToBlockTexture()
{
    glDeleteBuffers (1, &vbo_);

    glUseProgram(0);
    glDeleteProgram(shader_);
}


void ChunkToBlockTexture::
        mergeColumnMajorChunk(
                const Block& block,
                const Tfr::Chunk& chunk,
                BlockData& outData )
{
    mergeChunk( block, chunk );
}


void ChunkToBlockTexture::
        mergeRowMajorChunk(
                const Block& block,
                const Tfr::Chunk& chunk,
                BlockData& outData )
{
    mergeChunk( block, chunk );
}


void ChunkToBlockTexture::
        mergeChunk(
                const Block& block,
                const Tfr::Chunk& chunk )
{
    GlTexture::Ptr t = block.glblock->glTexture ();
    GlFrameBuffer fbo(t->getOpenGlTextureId ());

    GlFrameBuffer::ScopeBinding sb = fbo.getScopeBinding ();

    glPushAttribContext pa(GL_ENABLE_BIT);
    glDisable (GL_DEPTH_TEST);
    glDisable (GL_BLEND);
    glDisable (GL_CULL_FACE);

    GLint viewport[4];
    glGetIntegerv(GL_VIEWPORT, viewport);
    GlException_SAFE_CALL( glViewport(0, 0, fbo.getWidth (), fbo.getHeight () ) );

    glPushMatrixContext mpc( GL_PROJECTION );
    glLoadIdentity();
    Region br = block.getRegion ();
    glOrtho(br.a.time, br.b.time, br.a.scale, br.b.scale, -10,10);
    glPushMatrixContext mc( GL_MODELVIEW );
    glLoadIdentity();

    Tfr::FreqAxis display_scale = block.visualization_params ()->display_scale ();
    Region r = Region(Position(),Position());
    r.a.time = (chunk.chunk_offset/chunk.sample_rate).asFloat();
    r.b.time = r.a.time + chunk.nSamples ()/chunk.sample_rate;

    unsigned Y = chunk.nScales (), X = chunk.nSamples ();
    float min_hz = chunk.freqAxis.getFrequency (0u);
    float max_hz = chunk.freqAxis.getFrequency (Y);
    r.a.scale = display_scale.getFrequencyScalar (min_hz);
    r.b.scale = display_scale.getFrequencyScalar (max_hz);

    INFO TaskTimer tt(boost::format("Painting %s onto %s") % r % br);

    struct vertex_format {
        float x, y, u, v;
    };

    bool transpose = chunk.order == Tfr::Chunk::Order_column_major;
    float iY = 1.f/(Y-1);

    unsigned data_width  = transpose ? Y : X,
             data_height = transpose ? X : Y;
    GlTexture source( data_width, data_height, GL_RG, GL_RG, GL_FLOAT, chunk.transform_data->getCpuMemory ());

    std::vector<float> vertices((Y*2)*sizeof(vertex_format)/sizeof(float));
    int i=0;
    for (unsigned y=0; y<Y; y++) {
        float hz = chunk.freqAxis.getFrequency (y);
        float s = display_scale.getFrequencyScalar(hz);
        vertices[i++] = r.a.time;
        vertices[i++] = s;
        vertices[i++] = transpose ? y*iY : 0;
        vertices[i++] = transpose ? 0 : y*iY;
        vertices[i++] = r.b.time;
        vertices[i++] = s;
        vertices[i++] = transpose ? y*iY : 1;
        vertices[i++] = transpose ? 1 : y*iY;
    }

    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER, sizeof(float)*vertices.size (), &vertices[0], GL_STREAM_DRAW);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_TEXTURE_COORD_ARRAY);

    glVertexPointer(2, GL_FLOAT, sizeof(vertex_format), 0);
    glTexCoordPointer(2, GL_FLOAT, sizeof(vertex_format), (float*)0 + 2);

    GlTexture::ScopeBinding texObjBinding = source.getScopeBinding();
    glUseProgram(shader_);
    glUniform1f(normalization_location_, normalization_factor);
    glDrawArrays(GL_TRIANGLE_STRIP, 0, Y*2);
    glUseProgram(0);

    glDisableClientState(GL_TEXTURE_COORD_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    GlException_SAFE_CALL( glViewport(viewport[0], viewport[1], viewport[2], viewport[3] ) );

    PRINT_TEXTURES PRINT_DATASTORAGE(GlTextureRead(fbo.getGlTexture()).readFloat (), "fbo");
    PRINT_TEXTURES PRINT_DATASTORAGE(GlTextureRead(source.getOpenGlTextureId ()).readFloat (), "source");

    const_cast<Block*>(&block)->discard_new_block_data ();
}


} // namespace Heightmap



namespace Heightmap {

void ChunkToBlockTexture::
        test()
{
    // It should merge the contents of a chunk directly onto the texture of a block.
    {

    }
}

} // namespace Heightmap
