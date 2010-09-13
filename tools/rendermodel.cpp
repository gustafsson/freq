#include "rendermodel.h"

namespace Tools
{

RenderModel::
        RenderModel(pProject p)
        : _project(p)
{
    collection.reset( new Heightmap::Collection(_project->worker));
    collectionCallback.reset( new Signal::WorkerCallback( _project->worker, collection->postsink() ));
}

} // namespace Tools
