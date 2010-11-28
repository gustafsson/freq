#include "rendermodel.h"
#include "sawe/project.h"

namespace Tools
{

RenderModel::
        RenderModel(Sawe::Project* p)
        : _project(p),
        _qx(0), _qy(0), _qz(.5f), // _qz(3.6f/5),
        _px(0), _py(0), _pz(-10),
        _rx(91), _ry(180), _rz(0),
        xscale(1)
{
    Signal::BufferSource* bs = dynamic_cast<Signal::BufferSource*>(p->head_source()->root());
    BOOST_ASSERT(bs);
    collections.resize(bs->num_channels());
    for (unsigned c=0; c<bs->num_channels(); ++c)
    {
        collections[c].reset( new Heightmap::Collection(&_project->worker));
    }
    collection = collections[0];
    collectionCallback.reset( new Signal::WorkerCallback( &_project->worker, collection->postsink() ));

    renderer.reset( new Heightmap::Renderer( collection.get() ));
}

} // namespace Tools
