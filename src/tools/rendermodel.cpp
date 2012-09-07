#include "rendermodel.h"
#include "sawe/project.h"

#include "heightmap/collection.h"
#include "heightmap/renderer.h"

#include "tfr/filter.h"

#include <GlTexture.h>

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

    Signal::PostSink* o = renderSignalTarget->post_sink();

    BOOST_ASSERT( o->num_channels() );

    collections.resize(o->num_channels());
    for (unsigned c=0; c<o->num_channels(); ++c)
        collections[c].reset( new Heightmap::Collection(renderSignalTarget->source()));

    renderer.reset( new Heightmap::Renderer( collections[0].get() ));

    for (unsigned c=0; c<o->num_channels(); ++c)
        collections[c]->renderer = renderer.get();


//    setTestCamera();
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
    _qz = .5f;
    _px = 0;
    _py = 0;
    _pz = -10.f;
    _rx = 91;
    _ry = 180;
    _rz = 0;
    xscale = -_pz*0.1f;
    zscale = -_pz*0.75f;

#ifdef TARGET_hast
    _pz = -6;
    xscale = 0.1f;

    float L = _project->worker.length();
    if (L)
    {
        xscale = 14/L;
        _qx = 0.5*L;
    }
#endif
}


void RenderModel::
        setTestCamera()
{
    renderer->y_scale = 0.01f;
    _qx = 63.4565;
    _qy = 0;
    _qz = 0.37;
    _px = 0;
    _py = 0;
    _pz = -10;
    _rx = 46.2;
    _ry = 253.186;
    _rz = 0;

    orthoview.reset( _rx >= 90 );
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


float RenderModel::
        effective_ry()
{
    return fmod(fmod(_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(_ry,360)+360, 360)+45)/90))*orthoview;
}


} // namespace Tools
