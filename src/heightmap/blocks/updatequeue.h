#ifndef HEIGHTMAP_BLOCKS_UPDATEQUEUE_H
#define HEIGHTMAP_BLOCKS_UPDATEQUEUE_H

#include "heightmap/block.h"
#include "iupdatejob.h"

#include "blocking_queue.h"

#include <vector>

namespace Heightmap {
namespace Blocks {

namespace UpdateQueue {
    struct Job {
        IUpdateJob::ptr updatejob;
        std::vector<pBlock> intersecting_blocks;

        operator bool() const { return updatejob && !intersecting_blocks.empty (); }
    };

    typedef JustMisc::blocking_queue<Job>::queue queue;
    typedef std::shared_ptr<JustMisc::blocking_queue<Job>> ptr;
} // namespace UpdateQueue

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_UPDATEQUEUE_H
