#ifndef RENDERMODEL_H
#define RENDERMODEL_H

namespace Sawe {
    class Project;
}

#include "heightmap/collection.h"

namespace Tools
{
    class RenderModel
    {
    public:
        RenderModel(Sawe::Project* p);

        boost::scoped_ptr<Heightmap::Collection> collection;
        Signal::pWorkerCallback collectionCallback;

    private:
        friend class RenderView; // todo remove
        friend class RenderController; // todo remove
        Sawe::Project* project;
    };
} // namespace Tools

#endif // RENDERMODEL_H
