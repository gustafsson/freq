#include "recordmodel.h"

#include "adapters/microphonerecorder.h"
#include "sawe/project.h"

namespace Tools
{

RecordModel::
        RecordModel( Sawe::Project* project, RenderView* render_view )
    :
    project(project),
    render_view(render_view)
{
    recording = dynamic_cast<Adapters::MicrophoneRecorder*>
                (project->head_source()->root());

    BOOST_ASSERT( recording );
}


RecordModel::
        ~RecordModel()
{
    if (recording && !recording->isStopped())
        recording->stopRecording();
}


bool RecordModel::
        canCreateRecordModel( Sawe::Project* project )
{
    return dynamic_cast<Adapters::MicrophoneRecorder*>
                (project->head_source()->root());
}


} // namespace Tools
