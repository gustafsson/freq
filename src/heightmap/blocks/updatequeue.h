#ifndef HEIGHTMAP_BLOCKS_UPDATEQUEUE_H
#define HEIGHTMAP_BLOCKS_UPDATEQUEUE_H

#include "heightmap/block.h"
#include "iupdatejob.h"

#include "blocking_queue.h"

#include <vector>
#include <future>

namespace Heightmap {
namespace Blocks {

class UpdateQueue {
public:
    struct Job {
        IUpdateJob::ptr     updatejob;
        std::vector<pBlock> intersecting_blocks;
        std::promise<void>  promise;

        explicit operator bool() const;
    };

    typedef std::shared_ptr<UpdateQueue> ptr;
    typedef JustMisc::blocking_queue<Job>::abort_exception abort_exception;
    typedef JustMisc::blocking_queue<Job>::queue queue;


    std::future<void>   push (IUpdateJob::ptr updatejob, std::vector<pBlock> intersecting_blocks);
    Job                 pop ();
    queue               clear ();
    bool                empty ();
    void                abort_on_empty ();

private:
    JustMisc::blocking_queue<Job> q_;
};

} // namespace Blocks
} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKS_UPDATEQUEUE_H
