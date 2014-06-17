#include "rendermodel.h"
#include "sawe/project.h"

#include "heightmap/collection.h"
#include "heightmap/renderer.h"

#include "signal/operationwrapper.h"

#include "tfr/chunkfilter.h"

#include "support/renderoperation.h"

#include "GlTexture.h"

namespace Tools
{

class TargetInvalidator: public Signal::Processing::IInvalidator {
public:
    TargetInvalidator(Signal::Processing::TargetNeeds::const_ptr needs):needs_(needs) {}

    virtual void deprecateCache(Signal::Intervals what) const {
        Signal::Processing::TargetNeeds::deprecateCache(needs_, what);
    }

private:
    Signal::Processing::TargetNeeds::const_ptr needs_;
};

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
        transform_descs_(new Support::TransformDescs),
        stft_block_filter_params_(new Heightmap::TfrMappings::StftBlockFilterParams)

{
    Heightmap::BlockLayout bl(1<<8,1<<8,1);
    tfr_map_.reset (new Heightmap::TfrMapping(bl, 0));

    renderer.reset( new Heightmap::Renderer() );

    resetSettings();
//    setTestCamera();
}


RenderModel::
        ~RenderModel()
{
    TaskInfo ti(__FUNCTION__);
    target_marker_.reset ();
    render_operation_desc_.reset ();

    // Need to make sure that this thread really quits here, before the block cache is deleted.
    if (!block_update_queue)
        TaskInfo("!!! Lost block_update_queue");
    if (block_update_queue && !block_update_queue.unique ())
        TaskInfo("!!! block_update_queue unique");
    block_update_queue.reset ();

    renderer.reset();

    if (!tfr_map_)
        TaskInfo("!!! Lost tfr_map");
    if (tfr_map_ && !tfr_map_.unique ())
        TaskInfo("!!! tfr_map not unique");
    tfr_map_.reset ();
}


void RenderModel::
        init(Signal::Processing::Chain::ptr chain, Support::RenderOperationDesc::RenderTarget::ptr rt)
{
    // specify wrapped filter with set_filter
    Support::RenderOperationDesc*rod;
    render_operation_desc_.reset(rod=new Support::RenderOperationDesc(Signal::OperationDesc::ptr(), rt));
    target_marker_ = chain.write ()->addTarget(render_operation_desc_);
    rod->setInvalidator(Signal::Processing::IInvalidator::ptr(
                                               new TargetInvalidator(target_marker_->target_needs ())));
    chain_ = chain;

    recompute_extent ();
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
    renderer->render_settings.y_scale = 0.01f;
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


Heightmap::TfrMapping::Collections RenderModel::
        collections()
{
    return tfr_map_.read ()->collections();
}


void RenderModel::
        block_layout(Heightmap::BlockLayout bs)
{
    tfr_map_.write ()->block_layout( bs );
}


Tfr::FreqAxis RenderModel::
        display_scale()
{
    return tfr_map_.read ()->display_scale();
}


void RenderModel::
        display_scale(Tfr::FreqAxis x)
{
    if (x != display_scale ())
        if (block_update_queue) block_update_queue->clear();
    tfr_map_.write ()->display_scale( x );
}


Heightmap::AmplitudeAxis RenderModel::
        amplitude_axis()
{
    return tfr_map_.read ()->amplitude_axis();
}


void RenderModel::
        amplitude_axis(Heightmap::AmplitudeAxis x)
{
    if (x != amplitude_axis ())
        if (block_update_queue) block_update_queue->clear();
    tfr_map_.write ()->amplitude_axis( x );
}


Heightmap::TfrMapping::ptr RenderModel::
        tfr_mapping()
{
    return tfr_map_;
}


Support::TransformDescs::ptr RenderModel::
        transform_descs()
{
    return transform_descs_;
}


Tfr::TransformDesc::ptr RenderModel::
        transform_desc()
{
    auto o = render_operation_desc_.read ();
    const Support::RenderOperationDesc* rod = dynamic_cast<const Support::RenderOperationDesc*>(&*o);

    return rod
            ? rod->transform_desc ()
            : Tfr::TransformDesc::ptr();
}


void RenderModel::
        set_transform_desc(Tfr::TransformDesc::ptr t)
{
    {
        auto o = render_operation_desc_.write ();
        Support::RenderOperationDesc* rod = dynamic_cast<Support::RenderOperationDesc*>(&*o);

        if (!rod)
            return;

        rod->transform_desc (t);
    }

//    target_marker (.write ())->updateNeeds(
//                Signal::Intervals(),
//                Signal::Interval::IntervalType_MIN,
//                Signal::Interval::IntervalType_MAX,
//                Signal::Intervals::Intervals_ALL);
}


void RenderModel::
        recompute_extent()
{
    Signal::OperationDesc::Extent extent = chain_.read ()->extent(target_marker_);
    set_extent(extent);
}


void RenderModel::
        set_extent(Signal::OperationDesc::Extent extent)
{
    auto w = tfr_map_.write ();
    w->targetSampleRate( extent.sample_rate.get_value_or (1) );
    w->length( extent.interval.get_value_or (Signal::Interval()).count() / w->targetSampleRate() );
    w->channels( extent.number_of_channels.get_value_or (1) );
}


Signal::Processing::TargetMarker::ptr RenderModel::
        target_marker()
{
    return target_marker_;
}


void RenderModel::
        set_filter(Signal::OperationDesc::ptr o)
{
    auto wo = render_operation_desc_.write ();
    Signal::OperationDescWrapper* w =
            dynamic_cast<Signal::OperationDescWrapper*>(&*wo);

    w->setWrappedOperationDesc (o);
    wo.unlock ();

    Signal::Processing::IInvalidator::ptr i =
            render_operation_desc_.raw ()->getInvalidator ();
    if (i)
        i->deprecateCache(Signal::Interval::Interval_ALL);
}


Signal::OperationDesc::ptr RenderModel::
        get_filter()
{
    auto ow = render_operation_desc_.read ();
    const Signal::OperationDescWrapper* w = dynamic_cast<const Signal::OperationDescWrapper*>(&*ow);
    return w->getWrappedOperationDesc ();
}


Heightmap::TfrMappings::StftBlockFilterParams::ptr RenderModel::
        get_stft_block_filter_params()
{
    return stft_block_filter_params_;
}


float RenderModel::
        effective_ry()
{
    return fmod(fmod(_ry,360)+360, 360) * (1-orthoview) + (90*(int)((fmod(fmod(_ry,360)+360, 360)+45)/90))*orthoview;
}


} // namespace Tools
