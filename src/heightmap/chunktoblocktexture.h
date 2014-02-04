#ifndef HEIGHTMAP_CHUNKTOBLOCKTEXTURE_H
#define HEIGHTMAP_CHUNKTOBLOCKTEXTURE_H

#include "block.h"
#include "tfr/chunkfilter.h"
#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"

#include <boost/noncopyable.hpp>

namespace Heightmap {

/**
 * @brief The ChunkToBlockTexture class should merge the contents of a chunk
 * directly onto the texture of a block.
 */
class ChunkToBlockTexture: public boost::noncopyable
{
public:
    ChunkToBlockTexture();
    ~ChunkToBlockTexture();

    ComplexInfo complex_info;
    float normalization_factor;
    bool full_resolution;
    bool enable_subtexel_aggregation;

    void mergeColumnMajorChunk(
            const Block& block,
            const Tfr::Chunk&,
            BlockData& outData );

    void mergeRowMajorChunk(
            const Block& block,
            const Tfr::Chunk&,
            BlockData& outData );

    void mergeChunk(
            const Block& block,
            const Tfr::Chunk& );

private:
    unsigned vbo_;
    unsigned shader_;
    unsigned normalization_location_;
    unsigned amplitude_axis_location_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_CHUNKTOBLOCKTEXTURE_H
