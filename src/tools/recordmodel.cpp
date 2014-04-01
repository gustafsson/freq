#include "recordmodel.h"

#include "adapters/recorder.h"
#include "adapters/microphonerecorder.h"
#include "sawe/project.h"

#include <QSemaphore>

namespace Tools
{

RecordModel::
        RecordModel( Sawe::Project* project, RenderView* render_view, Adapters::Recorder::Ptr recording )
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
    if (recording) {
        auto w = recording.write ();
        if (!w->isStopped())
            w->stopRecording();
    }
}


class GotDataCallback: public Adapters::Recorder::IGotDataCallback
{
public:
    void setInvalidator(Signal::Processing::IInvalidator::Ptr i) { i_ = i; }
    void setRecordModel(RecordModel* model) { model_ = model; }

    virtual void markNewlyRecordedData(Signal::Interval what) {
        if (i_)
            i_.read ()->deprecateCache(what);
        if (model_)
            emit model_->markNewlyRecordedData(what);
    }

private:
    Signal::Processing::IInvalidator::Ptr i_;
    RecordModel* model_ = 0;
};


RecordModel* RecordModel::
        createRecorder(Signal::Processing::Chain::Ptr chain, Signal::Processing::TargetMarker::Ptr at,
                       Adapters::Recorder::Ptr recorder,
                       Sawe::Project* project, RenderView* render_view)
{
    Adapters::Recorder::IGotDataCallback::Ptr callback(new GotDataCallback());

    Signal::OperationDesc::Ptr desc( new Adapters::MicrophoneRecorderDesc(recorder, callback) );
    Signal::Processing::IInvalidator::Ptr i = chain.write ()->addOperationAt(desc, at);

    RecordModel* record_model = new RecordModel(project, render_view, recorder);
    record_model->recorder_desc = desc;
    record_model->invalidator = i;

    dynamic_cast<GotDataCallback*>(&*callback.write ())->setInvalidator (i);
    dynamic_cast<GotDataCallback*>(&*callback.write ())->setRecordModel (record_model);

    return record_model;
}


bool RecordModel::
        canCreateRecordModel( Sawe::Project* )
{
    return Adapters::MicrophoneRecorder(-1).canRecord ();
}

} // namespace Tools

#include <QApplication>
#include "signal/processing/workers.h"

namespace Tools
{

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
    std::string name = "RecordModel";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);

    // It should describe the operation required to perform a recording.
    {
        QSemaphore semaphore;
        Signal::Processing::Chain::Ptr chain = Signal::Processing::Chain::createDefaultChain ();
        Signal::OperationDesc::Ptr target_desc( new TargetMockDesc(&semaphore));
        Sawe::Project* p = (Sawe::Project*)1;
        Tools::RenderView* r = (Tools::RenderView*)2;


        Signal::Processing::TargetMarker::Ptr target_marker = chain.write ()->addTarget(target_desc);
        Signal::Processing::Step::Ptr step = target_marker->step().lock();

        RecordModel* record_model = RecordModel::createRecorder(
                    chain,
                    target_marker,
                    Adapters::Recorder::Ptr(new Adapters::MicrophoneRecorder(-1)),
                    p, r );

        EXCEPTION_ASSERT(record_model->recording);
        EXCEPTION_ASSERT_EQUALS(record_model->project, p);
        EXCEPTION_ASSERT_EQUALS(record_model->render_view, r);

        EXCEPTION_ASSERT_EQUALS(step.read ()->out_of_date(), ~Signal::Intervals());

        Signal::Processing::TargetNeeds::Ptr needs = target_marker->target_needs();
        needs.write ()->updateNeeds(Signal::Intervals(10,20));

        Signal::OperationDesc::Extent x = chain.read ()->extent(target_marker);
        EXCEPTION_ASSERT_EQUALS(x.interval.get_value_or (Signal::Interval(-1,0)), Signal::Interval());

        // Wait for the chain workers to finish fulfilling the target needs
        if (!Signal::Processing::TargetNeeds::sleep(needs, 1000)) {
            auto w = chain.read ()->workers().write();
            Signal::Processing::Workers::print(w->clean_dead_workers());
            EXCEPTION_ASSERT( false );
        }
        EXCEPTION_ASSERT_EQUALS(step.read ()->out_of_date(), ~Signal::Intervals(10,20));

        semaphore.acquire (semaphore.available ());
        record_model->recording.write ()->startRecording ();

        // Wait for the recorder to produce data within 1 second
        EXCEPTION_ASSERT(semaphore.tryAcquire (1, 1000));

        x = chain.read ()->extent(target_marker);
        EXCEPTION_ASSERT_LESS(400u, x.interval.get_value_or (Signal::Interval()).count());
    }
}


} // namespace Tools
