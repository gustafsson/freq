#include "rendermodel.h"
#include "sawe/project.h"

namespace Tools
{

RenderModel::
        RenderModel(Sawe::Project* p)
        : project(p)
{
    collection.reset( new Heightmap::Collection(&project->worker));
    collectionCallback.reset( new Signal::WorkerCallback( &project->worker, collection->postsink() ));

    renderer.reset( new Heightmap::Renderer( collection.get() ));
}

} // namespace Tools
