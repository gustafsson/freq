#ifndef HEIGHTMAP_UPDATE_OPENGL_SOURCE2PBO_H
#define HEIGHTMAP_UPDATE_OPENGL_SOURCE2PBO_H

#include "zero_on_move.h"
#include "tfr/chunk.h"

#include <future>

namespace Heightmap {
namespace Update {
namespace OpenGL {

class Source2Pbo
{
public:
    Source2Pbo(Tfr::pChunk chunk);
    Source2Pbo(Source2Pbo&& b) = default;
    Source2Pbo(Source2Pbo&) = delete;
    Source2Pbo& operator=(Source2Pbo&) = delete;
    ~Source2Pbo();

    std::packaged_task<void()> transferData(float *p);

    // Assumes transferData has been called first, will hang here otherwise.
    unsigned getPboWhenReady() { finishTransfer(); return chunk_pbo_; }

private:
    void setupPbo();
    void finishTransfer();

    const Tfr::ChunkData::ptr chunk_;
    const int n;
    JustMisc::zero_on_move<float*> mapped_chunk_data_;
    JustMisc::zero_on_move<unsigned> chunk_pbo_;
    std::future<void> data_transfer;
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_OPENGL_SOURCE2PBO_H
