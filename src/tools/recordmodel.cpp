#include "recordmodel.h"

#include "adapters/recorder.h"
#include "sawe/project.h"

namespace Tools
{

RecordModel::
        RecordModel( Sawe::Project* project, RenderView* render_view )
    :
    project(project),
    render_view(render_view)
{
    recording = dynamic_cast<Adapters::Recorder*>
                (project->head->head_source()->root());

    EXCEPTION_ASSERT( recording );
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
    return dynamic_cast<Adapters::Recorder*>
                (project->head->head_source()->root());
}


} // namespace Tools
