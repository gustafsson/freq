#include "rendermodel.h"
#include "sawe/project.h"

#include "heightmap/collection.h"
#include "heightmap/renderer.h"

#include "signal/operationwrapper.h"
#include "signal/oldoperationwrapper.h"

#include "tfr/filter.h"

#include <GlTexture.h>

namespace Tools
{



RenderModel::
        RenderModel(Sawe::Project* p)
        :
        //renderSignalTarget(new Signal::Target(&p->layers, "Heightmap", true, true)),
        _qx(0), _qy(0), _qz(0),
        _px(0), _py(0), _pz(0),
        _rx(0), _ry(0), _rz(0),
        xscale(0),
        zscale(0),
        orthoview(1),
        _project(p),
        transform_descs_(new Support::TransformDescs)
{
    Signal::OperationDescWrapper::Ptr w(new Signal::OperationDescWrapper());
    target_marker_ = write1(p->processing_chain ())->addTarget(w);

    resetSettings();

    // initialize tfr_map_
    //Signal::PostSink* o = renderSignalTarget->post_sink();
    //EXCEPTION_ASSERT_LESS( 0, o->num_channels () );

    Heightmap::BlockSize bs(1<<8,1<<8);
    tfr_map_.reset (new Heightmap::TfrMap(Heightmap::TfrMapping(bs,1), 0));

    recompute_extent ();

    renderer.reset( new Heightmap::Renderer() );

//    setTestCamera();
}


RenderModel::
        ~RenderModel()
{
    TaskInfo ti(__FUNCTION__);
    renderer.reset();
    tfr_map_.reset ();
    //renderSignalTarget.reset();
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


Heightmap::TfrMap::Collections RenderModel::
        collections()
{
    return read1(tfr_map_)->collections();
}


void RenderModel::
        block_size(Heightmap::BlockSize bs)
{
    write1(tfr_map_)->block_size( bs );
}


Tfr::FreqAxis RenderModel::
        display_scale()
{
    return read1(tfr_map_)->display_scale();
}


void RenderModel::
        display_scale(Tfr::FreqAxis x)
{
    write1(tfr_map_)->display_scale( x );
}


Heightmap::AmplitudeAxis RenderModel::
        amplitude_axis()
{
    return read1(tfr_map_)->amplitude_axis();
}


void RenderModel::
        amplitude_axis(Heightmap::AmplitudeAxis x)
{
    write1(tfr_map_)->amplitude_axis( x );
}


Heightmap::TfrMapping RenderModel::
        tfr_mapping()
{
    return read1(tfr_map_)->tfr_mapping();
}


Heightmap::TfrMap::Ptr RenderModel::
        tfr_map()
{
    return tfr_map_;
}


Support::TransformDescs::Ptr RenderModel::
        transform_descs()
{
    return transform_descs_;
}


Tfr::Filter* RenderModel::
        block_filter()
{
    EXCEPTION_ASSERTX(false, "Use Signal::Processing namespace");
/*
    std::vector<Signal::pOperation> s = renderSignalTarget->post_sink ()->sinks ();
    Tfr::Filter* f = dynamic_cast<Tfr::Filter*>(s[0]->source().get());

    return f;
*/
    return 0;
}


Tfr::TransformDesc::Ptr RenderModel::
        transform_desc()
{
    Signal::OperationDesc::Ptr o = get_filter();

    Tfr::FilterDesc* f = dynamic_cast<Tfr::FilterDesc*>(o.get ());
    if (f)
        return f->transformDesc ();

//Use Signal::Processing namespace
    Signal::OperationDescWrapper* dw = dynamic_cast<Signal::OperationDescWrapper*>(o.get());
    Signal::OldOperationDescWrapper* w = dynamic_cast<Signal::OldOperationDescWrapper*>(dw?dw->getWrappedOperationDesc ().get ():0);
    if (w)
    {
        Tfr::Filter* f2 = dynamic_cast<Tfr::Filter*>(w->old_operation ().get ());
        if (f2)
            return f2->transform ()->transformDesc ()->copy ();
    }
    return Tfr::TransformDesc::Ptr();
}


void RenderModel::
        set_transform_desc(Tfr::TransformDesc::Ptr t)
{
    Signal::OperationDesc::Ptr o = get_filter();
    Tfr::FilterDesc* f = dynamic_cast<Tfr::FilterDesc*>(o.get ());
    if (f)
        f->transformDesc (t);

//Use Signal::Processing namespace
    Signal::OperationDescWrapper* dw = dynamic_cast<Signal::OperationDescWrapper*>(o.get());
    Signal::OldOperationDescWrapper* w = dynamic_cast<Signal::OldOperationDescWrapper*>(dw?dw->getWrappedOperationDesc ().get ():0);
    if (w)
    {
        Tfr::Filter* f2 = dynamic_cast<Tfr::Filter*>(w->old_operation ().get ());
        if (f2)
            return f2->transform (t->createTransform ());
    }
}


void RenderModel::
        recompute_extent()
{
    Signal::OperationDesc::Extent extent = read1(project ()->processing_chain ())->extent(target_marker_);

    Heightmap::TfrMap::WritePtr w(tfr_map_);
    w->targetSampleRate( extent.sample_rate.get_value_or (1) );
    w->length( extent.interval.get_value_or (Signal::Interval()).count() / w->targetSampleRate() );
    w->channels( extent.number_of_channels.get_value_or (1) );
}


Signal::Processing::TargetMarker::Ptr RenderModel::
        target_marker()
{
    return target_marker_;
}


void RenderModel::
        set_filter(Signal::OperationDesc::Ptr o)
{
    Signal::Processing::Step::Ptr s = read1(target_marker_)->step().lock();
    Signal::OperationDesc::Ptr od = read1(s)->operation_desc();
    Signal::OperationDescWrapper* w = dynamic_cast<Signal::OperationDescWrapper*>(od.get());
    w->setWrappedOperationDesc (o);
}


Signal::OperationDesc::Ptr RenderModel::
        get_filter()
{
    Signal::Processing::Step::Ptr s = read1(target_marker_)->step().lock();
    Signal::OperationDesc::Ptr od = read1(s)->operation_desc();
    Signal::OperationDescWrapper* w = dynamic_cast<Signal::OperationDescWrapper*>(od.get());
    Signal::OperationDesc::Ptr o = w->getWrappedOperationDesc ();
    return o;
}


float RenderModel::
        effective_ry()
{
    return fmod(fmod(_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(_ry,360)+360, 360)+45)/90))*orthoview;
}


} // namespace Tools
