#ifndef RECORDMODEL_H
#define RECORDMODEL_H

#include "signal/operation.h"
#include "signal/processing/chain.h"
#include "adapters/recorder.h"

namespace Sawe { class Project; }

namespace Tools
{

class RenderView;

/**
 * @brief The RecordModel class should describe the operation required to perform a recording.
 *
 * Note that it's up to each renderview to update the stuff that it needs.
 * RecordModel doesn't concern itself with the framerate of RenderView.
 */
class RecordModel
{
public:
    /**
     * @brief createRecorder returns a new MicrophoneRecorder operation
     * description that can be added to a signal processing chain.
     * @return a new RecordModel if it could be created, or null if it failed.
     */
    static RecordModel* createRecorder( Signal::Processing::Chain::Ptr chain, Signal::Processing::TargetMarker::Ptr at,
                                 Adapters::Recorder::Ptr recorder, Sawe::Project* project, RenderView* render_view );
    ~RecordModel();

    static bool canCreateRecordModel( Sawe::Project* project );


    Adapters::Recorder::Ptr recording;
    Signal::Processing::IInvalidator::Ptr invalidator;
    Sawe::Project* project;
    RenderView* render_view;
    Signal::OperationDesc::Ptr recorderDesc() { return recorder_desc; }

private:
    RecordModel( Sawe::Project* project, RenderView* render_view, Adapters::Recorder::Ptr recording );

    Signal::OperationDesc::Ptr recorder_desc;

public:
    static void test();
};

} // namespace Tools

#endif // RECORDMODEL_H
