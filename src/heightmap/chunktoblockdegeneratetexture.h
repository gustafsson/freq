#ifndef HEIGHTMAP_CHUNKTOBLOCKDEGENERATETEXTURE_H
#define HEIGHTMAP_CHUNKTOBLOCKDEGENERATETEXTURE_H

#include "block.h"
#include "tfr/chunkfilter.h"
#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"
#include "ichunktoblock.h"

class GlTexture;

namespace Heightmap {

/**
 * @brief The ChunkToBlockDegenerateTexture class should merge the contents of
 * a chunk directly onto the texture of a block.
 */
class ChunkToBlockDegenerateTexture: public IChunkToBlock
{
public:
    ChunkToBlockDegenerateTexture(Tfr::pChunk chunk);
    ~ChunkToBlockDegenerateTexture();

    void mergeChunk(pBlock block);

private:
    void prepVbo(Tfr::FreqAxis display_scale);

    std::shared_ptr<GlTexture> chunk_texture_;
    Tfr::FreqAxis display_scale;
    Tfr::FreqAxis chunk_scale;
    float a_t, b_t, u0, u1;
    unsigned nScales, nSamples, nValidSamples;
    bool transpose;

    unsigned vbo_;
    unsigned shader_;
    unsigned normalization_location_;
    unsigned amplitude_axis_location_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKTOBLOCKDEGENERATETEXTURE_H
