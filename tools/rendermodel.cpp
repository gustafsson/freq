#include "rendermodel.h"
#include "sawe/project.h"

namespace Tools
{

RenderModel::
        RenderModel(Sawe::Project* p)
        : _project(p)
{
    collection.reset( new Heightmap::Collection(&_project->worker));
    collectionCallback.reset( new Signal::WorkerCallback( &_project->worker, collection->postsink() ));

    renderer.reset( new Heightmap::Renderer( collection.get() ));
}

} // namespace Tools
