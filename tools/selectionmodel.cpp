#include "selectionmodel.h"
#include "sawe/project.h"

namespace Tools
{

SelectionModel::
        SelectionModel(Sawe::Project* p)
            : project(p)
{
    postsinkCallback.reset( new Signal::WorkerCallback( &p->worker, Signal::pOperation(new Signal::PostSink)) );

    float l = project->worker.source()->length();
    selection[0].x = l*.5f;
    selection[0].y = 0;
    selection[0].z = .85f;
    selection[1].x = l*sqrt(2.0f);
    selection[1].y = 0;
    selection[1].z = 2;

    // no selection
    selection[0].x = selection[1].x;
    selection[0].z = selection[1].z;
}

Signal::PostSink* SelectionModel::
        getPostSink()
{
    return dynamic_cast<Signal::PostSink*>(postsinkCallback->sink().get());
}

} // namespace Tools
