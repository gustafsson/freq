#ifndef RENDERMODEL_H
#define RENDERMODEL_H

#include "tfr/freqaxis.h"
#include "signal/worker.h"

#include <boost/serialization/nvp.hpp>

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
        double _qx, _qy, _qz; // camera focus point, i.e (10, 0, 0.5)
        float _px, _py, _pz, // camera position relative center, i.e (0, 0, -6)
            _rx, _ry, _rz; // rotatino around center
        float xscale;
        float zscale;

    private:
        friend class RenderView; // todo remove
        friend class RenderController; // todo remove
        friend class TimelineController; // todo remove
        Sawe::Project* _project; // project should probably be a member of RenderController instead

        friend class boost::serialization::access;
        RenderModel() { BOOST_ASSERT( false ); } // required by serialization, should never be called
        template<class Archive> void serialize(Archive& ar, const unsigned int /*version*/) {
            TaskInfo ti("%s", __FUNCTION__);
            ar
                    & BOOST_SERIALIZATION_NVP(_qx)
                    & BOOST_SERIALIZATION_NVP(_qy)
                    & BOOST_SERIALIZATION_NVP(_qz)
                    & BOOST_SERIALIZATION_NVP(_px)
                    & BOOST_SERIALIZATION_NVP(_py)
                    & BOOST_SERIALIZATION_NVP(_pz)
                    & BOOST_SERIALIZATION_NVP(_rx)
                    & BOOST_SERIALIZATION_NVP(_ry)
                    & BOOST_SERIALIZATION_NVP(_rz)
                    & BOOST_SERIALIZATION_NVP(xscale)
                    & BOOST_SERIALIZATION_NVP(zscale);

        }
    };
} // namespace Tools

#endif // RENDERMODEL_H
