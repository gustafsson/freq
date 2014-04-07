#ifndef HEIGHTMAP_BLOCKS_UPDATEQUEUE_H
#define HEIGHTMAP_BLOCKS_UPDATEQUEUE_H

#include "heightmap/mergechunk.h"
#include "shared_state.h"
#include "iupdatejob.h"

#include <queue>
#include <condition_variable>

namespace Heightmap {
namespace Blocks {

class UpdateQueue
{
public:
    typedef std::shared_ptr<UpdateQueue> ptr;

    struct Job {
        const IUpdateJob::ptr updatejob;
        const std::vector<pBlock> intersecting_blocks;

        operator bool() const { return updatejob && !intersecting_blocks.empty (); }
    };

    UpdateQueue();
    ~UpdateQueue();
    UpdateQueue(const UpdateQueue& b) = delete;
    UpdateQueue& operator=(const UpdateQueue& b) = delete;

    void clear ();

    // TODO assume constant MergeChunk::ptr merge_chunk
    void addJob (Job job);

    // Wait until there is any Job, return an empty Job if timeout elapsed or clear or abortGetJob was called.
    // timeout <= 0 returns immediately
    Job getJob(double timeout);

    // Wait until there is any Job, return an empty Job if clear or abortGetJob was called.
    Job getJob();

    bool isEmpty();

private:
    std::condition_variable_any got_chunk;

    struct jobqueue : public std::queue<const Job> {
        // Has only simple accessors, a simple mutex is faster than a more complex one
        typedef shared_state_mutex_notimeout_noshared shared_state_mutex;
    };

    shared_state<jobqueue> jobs;
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_UPDATEQUEUE_H
