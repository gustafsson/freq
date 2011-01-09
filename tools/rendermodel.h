#ifndef RENDERMODEL_H
#define RENDERMODEL_H

#include "tfr/freqaxis.h"
#include "signal/worker.h"

namespace Sawe {
    class Project;
}

namespace Heightmap
{
    class Collection;
    class Renderer;
}

namespace Tools
{
    class RenderModel
    {
    public:
        RenderModel(Sawe::Project* p);
        ~RenderModel();

        std::vector<boost::shared_ptr<Heightmap::Collection> > collections;

        Signal::pOperation postsink();
        Tfr::FreqAxis display_scale();

        Signal::pWorkerCallback collectionCallback;
        boost::shared_ptr<Heightmap::Renderer> renderer;

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
