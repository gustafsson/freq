#include "worker.h"

namespace Signal {
namespace Processing {

Worker::
        Worker (Signal::ComputingEngine::Ptr computing_eninge, GetTask::Ptr get_task)
    :
      computing_eninge_(computing_eninge),
      get_task_(get_task)
{

}


void Worker::
        run()
{
    Task::Ptr task;

    while (task = get_task_->getTask())
    {
        task->run(computing_eninge_);
    }

    deleteLater ();
}


class GetTaskMock: public GetTask {
public:
    GetTaskMock() : get_task_count(0) {}

    int get_task_count;

    virtual Task::Ptr getTask() volatile {
        if (get_task_count++ < 1)
            return Task::Ptr();

        pBuffer b(new Buffer(Interval(60,70), 40, 7));
        Signal::OperationDesc::Ptr od(new BufferSource(b));
        Step::Ptr step (new Step(od, b->sample_rate (), b->number_of_channels ()));
        std::vector<Step::Ptr> children;
        Signal::Interval expected_output(-10,80);

        // No children, and no data
        return Task::Ptr( new Task(step, children, expected_output));
    }
};

void Worker::
        test()
{
    // It should run the next task as long as there is one
    {
        Signal::ComputingEngine::Ptr computing_eninge(new Signal::ComputingCpu);
        GetTask::Ptr gettask(new GetTaskMock());

        Worker worker(computing_eninge, gettask);
        worker.run ();
        EXCEPTION_ASSERT_EQUALS( true, worker.isRunning () );
        QThread::currentThread()->wait (1);
        EXCEPTION_ASSERT_EQUALS( false, worker.isRunning () );
        EXCEPTION_ASSERT_EQUALS( 1, dynamic_cast<volatile GetTaskMock*>(gettask.get())->get_task_count );
    }
}


} // namespace Processing
} // namespace Signal
