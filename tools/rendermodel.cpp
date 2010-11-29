#include "rendermodel.h"
#include "sawe/project.h"

namespace Tools
{

RenderModel::
        RenderModel(Sawe::Project* p)
        :
        _qx(0), _qy(0), _qz(.5f), // _qz(3.6f/5),
        _px(0), _py(0), _pz(-10),
        _rx(91), _ry(180), _rz(0),
        xscale(1),
        _project(p)
{
	Signal::Operation* o = p->head_source()->root();
	Signal::FinalSource* fs = dynamic_cast<Signal::FinalSource*>(o);
	BOOST_ASSERT(fs);

    collections.resize(fs->num_channels());
    for (unsigned c=0; c<fs->num_channels(); ++c)
    {
        collections[c].reset( new Heightmap::Collection(&_project->worker));
        if (0<c)
            collections[c]->setPostsink( collections[0]->postsink() );
    }

    collectionCallback.reset( new Signal::WorkerCallback( &_project->worker, postsink() ));

    renderer.reset( new Heightmap::Renderer( collections[0].get() ));
}

} // namespace Tools
