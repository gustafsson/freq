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

        std::vector<boost::shared_ptr<Heightmap::Collection> > collections;
        // TODO remove
        boost::shared_ptr<Heightmap::Collection> collection;

        Signal::pWorkerCallback collectionCallback;
        Heightmap::pRenderer renderer;

        Sawe::Project* project() { return _project; }

        // TODO remove position and use renderer->camera instead
        double _qx, _qy, _qz; // position
        float _px, _py, _pz, // TODO beautify
            _rx, _ry, _rz;
        float xscale;

    private:
        friend class RenderView; // todo remove
        friend class RenderController; // todo remove
        friend class TimelineController; // todo remove
        Sawe::Project* _project; // project should probably be a member of RenderController instead
    };
} // namespace Tools

#endif // RENDERMODEL_H
