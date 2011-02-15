#include "rendermodel.h"
#include "sawe/project.h"

#include "heightmap/renderer.h"

namespace Tools
{

RenderModel::
        RenderModel(Sawe::Project* p)
        :
        _qx(0), _qy(0), _qz(.5f), // _qz(3.6f/5),
        _px(0), _py(0), _pz(-10),
        _rx(91), _ry(180), _rz(0),
        xscale(1),
        zscale(5),
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

#ifdef TARGET_sss
    _pz = -6;
    xscale = 0.1f;

    float L = p->worker.length();
    if (L)
    {
        xscale = 14/L;
        _qx = 0.5*L;
    }

    renderer->left_handed_axes = false;
#endif
}


RenderModel::
        ~RenderModel()
{
    TaskInfo ti(__FUNCTION__);
    Signal::PostSink* ps = dynamic_cast<Signal::PostSink*>(postsink().get());
    std::vector<Signal::pOperation> empty;
    ps->sinks(empty);

    collections.clear();
}


Signal::pOperation RenderModel::
        postsink()
{
    return collections[0]->postsink();
}


Tfr::FreqAxis RenderModel::
        display_scale()
{
    return collections[0]->display_scale();
}

} // namespace Tools
