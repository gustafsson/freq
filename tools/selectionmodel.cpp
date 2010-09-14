#include "sawe/project.h"

namespace Tools
{

SelectionModel::
        SelectionModel(Sawe::Project* p)
            : project(p)
{
    postsinkCallback.reset( new Signal::WorkerCallback( &p->worker, Signal::pOperation(new Signal::PostSink)) );
}

Signal::PostSink* SelectionModel::
        getPostSink()
{
    return dynamic_cast<Signal::PostSink*>(
            project->tools.selection_model.postsinkCallback->sink().get());
}

} // namespace Tools
