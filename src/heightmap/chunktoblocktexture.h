#ifndef HEIGHTMAP_CHUNKTOBLOCKTEXTURE_H
#define HEIGHTMAP_CHUNKTOBLOCKTEXTURE_H

#include "block.h"
#include "tfr/chunkfilter.h"
#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"
#include "ichunktoblock.h"

class GlTexture;

namespace Heightmap {

/**
 * @brief The ChunkToBlockTexture class should merge the contents of a chunk
 * directly onto the texture of a block.
 */
class ChunkToBlockTexture: public IChunkToBlock
{
public:
    ChunkToBlockTexture(Tfr::pChunk chunk);
    ~ChunkToBlockTexture();

    ComplexInfo complex_info;
    float normalization_factor;
    bool full_resolution;
    bool enable_subtexel_aggregation;

    void mergeChunk(pBlock block);

private:
    void prepVbo(Tfr::FreqAxis display_scale);

    std::shared_ptr<GlTexture> chunk_texture_;
    Tfr::FreqAxis display_scale;
    Tfr::FreqAxis chunk_scale;
    float a_t, b_t;
    unsigned nScales, nSamples;
    bool transpose;

    unsigned vbo_;
    unsigned shader_;
    unsigned normalization_location_;
    unsigned amplitude_axis_location_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKTOBLOCKTEXTURE_H
