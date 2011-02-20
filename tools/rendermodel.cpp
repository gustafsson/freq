#include "rendermodel.h"
#include "sawe/project.h"

#include "heightmap/renderer.h"

#include "tfr/filter.h"

namespace Tools
{



RenderModel::
        RenderModel(Sawe::Project* p)
        :
        renderSignalTarget(new Signal::Target(&p->layers, "Heightmap")),
        _qx(0), _qy(0), _qz(.5f), // _qz(3.6f/5),
        _px(0), _py(0), _pz(-10),
        _rx(91), _ry(180), _rz(0),
        xscale(1),
        zscale(5),
        _project(p)
{
    p->worker.target( renderSignalTarget );

    Signal::PostSink* o = renderSignalTarget->post_sink();


    BOOST_ASSERT( o->num_channels() );

    collections.resize(o->num_channels());
    for (unsigned c=0; c<o->num_channels(); ++c)
        collections[c].reset( new Heightmap::Collection(&_project->worker));

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
    renderer.reset();
    collections.clear();
    renderSignalTarget.reset();
}


Tfr::FreqAxis RenderModel::
        display_scale()
{
    return collections[0]->display_scale();
}


Tfr::Filter* RenderModel::
        block_filter()
{
    return dynamic_cast<Tfr::Filter*>(collections[0]->block_filter().get());
}

} // namespace Tools
