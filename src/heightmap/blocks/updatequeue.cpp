#include "updatequeue.h"
#include "tasktimer.h"

#include "tfr/chunk.h"

//#define INFO
#define INFO if(0)

namespace Heightmap {
namespace Blocks {

UpdateQueue::
        UpdateQueue()
    :
      jobs(new jobqueue)
{
}


UpdateQueue::
        ~UpdateQueue()
{
}


void UpdateQueue::
        clear()
{
    INFO TaskTimer ti("UpdateQueue::clear");

    auto jobs = this->jobs;

    while(!jobs->empty ())
        jobs->pop ();

    got_chunk.notify_all ();
}


void UpdateQueue::
        addJob( Job job )
{
    EXCEPTION_ASSERT( !job.intersecting_blocks.empty () );
    EXCEPTION_ASSERT( job.updatejob );

    jobs->push (job);

    got_chunk.notify_one ();
}


UpdateQueue::Job UpdateQueue::
        getJob(double timeout)
{
    auto jobs = this->jobs.write ();

    if (jobs->empty () && 0 < timeout)
        got_chunk.wait_for(jobs, std::chrono::duration<double>{timeout});

    if (jobs->empty())
    {
        return Job {};
    }
    else
    {
        Job j = jobs->front ();
        jobs->pop ();
        return j;
    }
}


UpdateQueue::Job UpdateQueue::
        getJob()
{
    auto jobs = this->jobs.write ();

    if (jobs->empty ())
        got_chunk.wait(jobs);

    if (jobs->empty())
    {
        return Job {};
    }
    else
    {
        Job j = jobs->front ();
        jobs->pop ();
        return j;
    }
}


bool UpdateQueue::
        isEmpty()
{
    return jobs->empty();
}


} // namespace Blocks
} // namespace Heightmap
