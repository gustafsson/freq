#ifndef RECORDMODEL_H
#define RECORDMODEL_H

#include <signal/operation.h>
#include "signal/processing/chain.h"

namespace Sawe { class Project; }
namespace Adapters { class Recorder; }

namespace Tools
{

class RenderView;

/**
 * @brief The RecordModel class should describe the operation required to perform a recording.
 */
class RecordModel
{
public:
    RecordModel( Sawe::Project* project, RenderView* render_view ); // TODO deprecated

    /**
     * @brief createRecorder returns a new MicrophoneRecorder operation
     * description that can be added to a signal processing chain.
     * @return a new RecordModel if it could be created, or null if it failed.
     */
    static RecordModel* createRecorder( Signal::Processing::Chain::Ptr chain, Signal::Processing::TargetMarker::Ptr at,
                                 Sawe::Project* project, RenderView* render_view );
    ~RecordModel();

    static bool canCreateRecordModel( Sawe::Project* project );


    Adapters::Recorder* recording;
    Sawe::Project* project;
    RenderView* render_view;

private:
    RecordModel( Sawe::Project* project, RenderView* render_view, Adapters::Recorder* recording );

    Signal::OperationDesc::Ptr recorder_desc;

public:
    static void test();
};

} // namespace Tools

#endif // RECORDMODEL_H
