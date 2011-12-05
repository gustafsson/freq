#include "rendermodel.h"
#include "sawe/project.h"

#include "heightmap/renderer.h"

#include "tfr/filter.h"

namespace Tools
{



RenderModel::
        RenderModel(Sawe::Project* p)
        :
        renderSignalTarget(new Signal::Target(&p->layers, "Heightmap", true, true)),
        _qx(0), _qy(0), _qz(0),
        _px(0), _py(0), _pz(0),
        _rx(0), _ry(0), _rz(0),
        xscale(0),
        zscale(0),
        orthoview(1),
        _project(p)
{
    resetSettings();

    p->worker.target( renderSignalTarget );

    Signal::PostSink* o = renderSignalTarget->post_sink();

    BOOST_ASSERT( o->num_channels() );

    collections.resize(o->num_channels());
    for (unsigned c=0; c<o->num_channels(); ++c)
        collections[c].reset( new Heightmap::Collection(renderSignalTarget->source()));

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


void RenderModel::
        resetSettings()
{
    _qx = 0;
    _qy = 0;
    _qz = .5f;  // _qz(3.6f/5),
    _px = 0;
    _py = 0;
    _pz = -40.f;
    _rx = 91;
    _ry = 180;
    _rz = 0;
    xscale = 3;
    zscale = -_pz*0.75f;
}


Tfr::FreqAxis RenderModel::
        display_scale()
{
    return collections[0]->display_scale();
}


void RenderModel::
        display_scale(Tfr::FreqAxis x)
{
    for (unsigned c=0; c<collections.size(); ++c)
        collections[c]->display_scale( x );
}


Heightmap::AmplitudeAxis RenderModel::
        amplitude_axis()
{
    return collections[0]->amplitude_axis();
}


void RenderModel::
        amplitude_axis(Heightmap::AmplitudeAxis x)
{
    for (unsigned c=0; c<collections.size(); ++c)
        collections[c]->amplitude_axis( x );
}


Tfr::Filter* RenderModel::
        block_filter()
{
    return dynamic_cast<Tfr::Filter*>(collections[0]->block_filter().get());
}

} // namespace Tools
