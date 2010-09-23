#ifndef RENDERMODEL_H
#define RENDERMODEL_H

namespace Sawe {
    class Project;
}

#include "heightmap/collection.h"
#include "heightmap/renderer.h"

namespace Tools
{
    class RenderModel
    {
    public:
        RenderModel(Sawe::Project* p);

        boost::scoped_ptr<Heightmap::Collection> collection;
        Signal::pWorkerCallback collectionCallback;
        Heightmap::pRenderer renderer;

    private:
        friend class RenderView; // todo remove
        friend class RenderController; // todo remove
        friend class TimelineController; // todo remove
        Sawe::Project* project; // project should probably be a member of RenderController instead
    };
} // namespace Tools

#endif // RENDERMODEL_H
