#include "recordmodel.h"

#include "adapters/recorder.h"
#include "adapters/microphonerecorder.h"
#include "sawe/project.h"

#include <QSemaphore>

namespace Tools
{

RecordModel::
        RecordModel( Sawe::Project* project, RenderView* render_view, Adapters::Recorder* recording )
    :
    recording(recording),
    project(project),
    render_view(render_view)
{
    EXCEPTION_ASSERT( recording );
}


RecordModel::
        ~RecordModel()
{
    if (recording && !recording->isStopped())
        recording->stopRecording();
}


class GotDataCallback: public Adapters::MicrophoneRecorderDesc::IGotDataCallback
{
public:
    void setInvalidator(Signal::Processing::IInvalidator::Ptr i) { i_ = i; }

    virtual void markNewlyRecordedData(Signal::Interval what) {
        if (i_)
            read1(i_)->deprecateCache(what);
    }

private:
    Signal::Processing::IInvalidator::Ptr i_;
};


RecordModel* RecordModel::
        createRecorder(Signal::Processing::Chain::Ptr chain, Signal::Processing::TargetMarker::Ptr at,
                       Adapters::Recorder* recorder,
                       Sawe::Project* project, RenderView* render_view)
{
    Adapters::MicrophoneRecorderDesc::IGotDataCallback::Ptr callback(new GotDataCallback());

    Signal::OperationDesc::Ptr desc( new Adapters::MicrophoneRecorderDesc(recorder, callback) );
    Signal::Processing::IInvalidator::Ptr i = write1(chain)->addOperationAt(desc, at);

    dynamic_cast<GotDataCallback*>(&*write1(callback))->setInvalidator (i);

    RecordModel* record_model = new RecordModel(project, render_view, recorder);
    record_model->recorder_desc = desc;
    return record_model;
}


bool RecordModel::
        canCreateRecordModel( Sawe::Project* )
{
    return Adapters::MicrophoneRecorder(-1).canRecord ();
}


class TargetMock: public Signal::Operation
{
public:
    TargetMock(QSemaphore* semaphore):semaphore_(semaphore) {}

private:
    virtual Signal::pBuffer process(Signal::pBuffer b) {
        semaphore_->release ();
        return b;
    }

    QSemaphore* semaphore_;
};

class TargetMockDesc: public Signal::OperationDesc
{
public:
    TargetMockDesc(QSemaphore* semaphore):semaphore_(semaphore) {}

private:
    virtual Signal::Interval requiredInterval( const Signal::Interval& I, Signal::Interval* expectedOutput ) const {
        if (expectedOutput)
            *expectedOutput = I;
        return I;
    }
    virtual Signal::Interval affectedInterval( const Signal::Interval& I ) const { return I; }
    virtual OperationDesc::Ptr copy() const {
        EXCEPTION_ASSERTX(false, "not implemented");
        return OperationDesc::Ptr();
    }
    virtual Signal::Operation::Ptr createOperation(Signal::ComputingEngine*) const {
        return Signal::Operation::Ptr(new TargetMock(semaphore_));
    }

    QSemaphore* semaphore_;
};

void RecordModel::
        test()
{
    // It should describe the operation required to perform a recording.
    {
        QSemaphore semaphore;
        Signal::Processing::Chain::Ptr chain = Signal::Processing::Chain::createDefaultChain ();
        Signal::OperationDesc::Ptr target_desc( new TargetMockDesc(&semaphore));
        Sawe::Project* p = (Sawe::Project*)1;
        Tools::RenderView* r = (Tools::RenderView*)2;


        Signal::Processing::TargetMarker::Ptr target_marker = write1(chain)->addTarget(target_desc);
        Signal::Processing::Step::Ptr step = target_marker->step().lock();

        RecordModel* record_model = RecordModel::createRecorder(
                    chain,
                    target_marker,
                    new Adapters::MicrophoneRecorder(-1),
                    p, r );

        EXCEPTION_ASSERT(record_model->recording);
        EXCEPTION_ASSERT_EQUALS(record_model->project, p);
        EXCEPTION_ASSERT_EQUALS(record_model->render_view, r);

        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), ~Signal::Intervals());

        Signal::Processing::TargetNeeds::Ptr needs = target_marker->target_needs();
        write1(needs)->updateNeeds(Signal::Intervals(10,20));

        Signal::OperationDesc::Extent x = read1(chain)->extent(target_marker);
        EXCEPTION_ASSERT_EQUALS(x.interval.get_value_or (Signal::Interval(-1,0)), Signal::Interval());

        // Wait for the chain workers to finish fulfilling the target needs
        EXCEPTION_ASSERT( needs->sleep(1000) );
        EXCEPTION_ASSERT_EQUALS(read1(step)->out_of_date(), ~Signal::Intervals(10,20));

        semaphore.acquire (semaphore.available ());
        record_model->recording->startRecording ();

        // Wait for the recorder to produce data within 1 second
        EXCEPTION_ASSERT(semaphore.tryAcquire (1, 1000));

        x = read1(chain)->extent(target_marker);
        EXCEPTION_ASSERT_LESS(400, x.interval.get_value_or (Signal::Interval()).count());
    }
}


} // namespace Tools
