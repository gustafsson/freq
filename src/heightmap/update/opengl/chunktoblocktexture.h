#ifndef HEIGHTMAP_CHUNKTOBLOCKTEXTURE_H
#define HEIGHTMAP_CHUNKTOBLOCKTEXTURE_H

#include "block.h"
#include "tfr/chunkfilter.h"
#include "tfr/freqaxis.h"
#include "heightmap/amplitudeaxis.h"

class GlTexture;

namespace Heightmap {
namespace Update {

/**
 * @brief The ChunkToBlockTexture class should merge the contents of a chunk
 * directly onto the texture of a block.
 *
 * TODO There is plenty of potential parallelism to exploit here.
 * 0) Measure what takes the most time.
 * 1) If multiple textures are to be created they can be uploaded asynchronously
 *    with VBOs first, and then call mergeChunk.
 * 2) Copying to VBO can be done from worker thread. Creating a texture from
 *    a VBO is fast. The VBO must be created in the GUI thread.
 */
class ChunkToBlockTexture
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
    float a_t, b_t, a_t0, b_t0;
    unsigned nScales, nSamples, nValidSamples;
    bool transpose;

    unsigned vbo_;
    unsigned shader_;
    unsigned normalization_location_;
    unsigned amplitude_axis_location_;

public:
    static void test();
};

} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKTOBLOCKTEXTURE_H
