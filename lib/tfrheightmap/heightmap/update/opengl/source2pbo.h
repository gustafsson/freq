#ifndef HEIGHTMAP_UPDATE_OPENGL_SOURCE2PBO_H
#define HEIGHTMAP_UPDATE_OPENGL_SOURCE2PBO_H

#include "zero_on_move.h"
#include "tfr/chunk.h"

#include <future>
#include <QOpenGLFunctions>

// PBOs are not supported on OpenGL ES (< 3.0)
#ifdef LEGACY_OPENGL
namespace Heightmap {
namespace Update {
namespace OpenGL {

class Source2Pbo: QOpenGLFunctions
{
public:
    Source2Pbo(Tfr::pChunk chunk, bool f32);
    Source2Pbo(Source2Pbo&& b) = default;
    Source2Pbo(Source2Pbo&) = delete;
    Source2Pbo& operator=(Source2Pbo&) = delete;
    ~Source2Pbo();

    std::packaged_task<void()> transferData(void *p);

    // Assumes transferData has been called first, will hang here otherwise.
    unsigned getPboWhenReady() { finishTransfer(); return chunk_pbo_; }
    bool f32() const { return f32_; }

private:
    void setupPbo();
    void finishTransfer();

    const Tfr::ChunkData::ptr chunk_;
    const int n;
    const bool f32_;
    JustMisc::zero_on_move<void*> mapped_chunk_data_;
    JustMisc::zero_on_move<unsigned> chunk_pbo_;
    std::future<void> data_transfer;
};

} // namespace OpenGL
} // namespace Update
} // namespace Heightmap
#endif

#endif // HEIGHTMAP_UPDATE_OPENGL_SOURCE2PBO_H
