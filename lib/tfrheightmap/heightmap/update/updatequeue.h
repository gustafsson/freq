#ifndef HEIGHTMAP_UPDATE_UPDATEQUEUE_H
#define HEIGHTMAP_UPDATE_UPDATEQUEUE_H

#include "heightmap/block.h"
#include "iupdatejob.h"

#include "blocking_queue.h"

#include <vector>
#include <future>

namespace Heightmap {
namespace Update {

class UpdateQueue {
public:
    struct Job {
        IUpdateJob::ptr     updatejob;
        std::vector<pBlock> intersecting_blocks;
        std::promise<void>  promise;

        explicit operator bool() const;
    };

    typedef std::shared_ptr<UpdateQueue> ptr;
    class skip_job_exception : public std::exception {};
    typedef JustMisc::blocking_queue<Job>::abort_exception abort_exception;
    typedef JustMisc::blocking_queue<Job>::queue queue;


    std::future<void>   push (IUpdateJob::ptr updatejob, std::vector<pBlock> intersecting_blocks);
    Job                 pop ();
    queue               clear ();
    bool                empty ();

    // this method is used to abort any blocking pop,
    // clear the queue, disable any future pops and
    // discard any future pushes
    void                close ();

private:
    JustMisc::blocking_queue<Job> q_;
};

} // namespace Update
} // namespace Heightmap

#endif // HEIGHTMAP_UPDATE_UPDATEQUEUE_H
