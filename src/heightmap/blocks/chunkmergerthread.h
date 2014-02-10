#ifndef HEIGHTMAP_BLOCKS_CHUNKMERGERTHREAD_H
#define HEIGHTMAP_BLOCKS_CHUNKMERGERTHREAD_H

#include "ichunkmerger.h"

#include <queue>
#include <QThread>
#include <QSemaphore>

class QGLWidget;

namespace Heightmap {
namespace Blocks {

/**
 * @brief The ChunkMergerThread class should update textures in a separate thread
 * from the worker thread.
 */
class ChunkMergerThread: public QThread, public IChunkMerger
{
    Q_OBJECT
public:
    typedef std::shared_ptr<ChunkMergerThread> Ptr;

    ChunkMergerThread(QGLWidget* shared_gl_context);
    ~ChunkMergerThread();

    // IChunkMerger
    void clear();
    void addChunk( MergeChunk::Ptr merge_chunk,
                   Tfr::ChunkAndInverse chunk,
                   std::vector<pBlock> intersecting_blocks );
    bool processChunks(float timeout) volatile;

    bool isEmpty() const;
    bool wait(float timeout) const;

private:
    struct Job {
        MergeChunk::Ptr merge_chunk;
        Tfr::ChunkAndInverse chunk;
        std::vector<pBlock> intersecting_blocks;
    };

    class Jobs: public VolatilePtr<Jobs>, public std::queue<Job> {};


    Jobs::Ptr   jobs;
    QSemaphore  semaphore;
    QGLWidget*  shared_gl_context;

    void        run();
    static void processJob(Job& j);

public:
    static void test();
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_CHUNKMERGERTHREAD_H