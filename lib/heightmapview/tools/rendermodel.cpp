#include "rendermodel.h"

#include "heightmap/collection.h"
#include "heightmap/render/renderer.h"
#include "signal/operationwrapper.h"

#include "tfr/chunkfilter.h"

#include "support/renderoperation.h"

#include "GlTexture.h"

namespace Tools
{

class TargetInvalidator: public Signal::Processing::IInvalidator {
public:
    TargetInvalidator(Signal::Processing::TargetNeeds::const_ptr needs) : needs_(needs) {}

    void deprecateCache(Signal::Intervals what) const override {
        needs_->deprecateCache(what);
    }

private:
    Signal::Processing::TargetNeeds::const_ptr needs_;
};

RenderModel::
        RenderModel()
        :
        transform_descs_(new Support::TransformDescs),
        stft_block_filter_params_(new Heightmap::TfrMappings::StftBlockFilterParams)
{
    Heightmap::BlockLayout bl(1<<8,1<<8,1);
    tfr_map_.reset (new Heightmap::TfrMapping(bl, 0));

    render_block.reset( new Heightmap::Render::RenderBlock(&render_settings));

    block_update_queue.reset (new Heightmap::Update::UpdateQueue::ptr::element_type());

    resetCameraSettings();
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

    render_block.reset();

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

    Signal::OperationDesc::Extent x = recompute_extent ();

    Heightmap::FreqAxis fa;
    fa.setLinear( x.sample_rate.get () );
    display_scale( fa );
}


void RenderModel::
        resetCameraSettings()
{
    camera.q = GLvector(0,0,.5f);
    camera.p = GLvector(0,0,-10.f);
    camera.r = GLvector(91,180,0);
    camera.xscale = -camera.p[2]*0.1f;
    camera.zscale = -camera.p[2]*0.75f;

    float L = tfr_mapping ().read ()->length();
    if (L)
    {
        camera.xscale = 10/L;
        camera.q[0] = 0.5*L;
    }

#ifdef TARGET_hast
    camera.p[2] = -6;
    xscale = 0.1f;

    if (L)
        camera.xscale = 14/L;
#endif
}


void RenderModel::
        resetBlockCaches()
{
    for (auto c : collections())
        c->cache()->clear();
}


void RenderModel::
        setTestCamera()
{
    render_settings.y_scale = 0.01f;
    camera.q = GLvector(63.4565,0,0.37);
    camera.p = GLvector(0,0,-10);
    camera.r = GLvector(46.2, 253.186, 0);

    camera.orthoview.reset( camera.r[0] >= 90 );
}


Heightmap::TfrMapping::Collections RenderModel::
        collections()
{
    return tfr_map_.read ()->collections();
}


Signal::Processing::Chain::ptr RenderModel::
        chain()
{
    return chain_;
}


void RenderModel::
        block_layout(Heightmap::BlockLayout bs)
{
    tfr_map_.write ()->block_layout( bs );
}


Heightmap::FreqAxis RenderModel::
        display_scale()
{
    return tfr_map_.read ()->display_scale();
}


void RenderModel::
        display_scale(Heightmap::FreqAxis x)
{
    if (x != display_scale ())
        if (block_update_queue)
            block_update_queue->clear();

    tfr_map_.write ()->display_scale( x );

    Signal::Processing::IInvalidator::ptr i =
            render_operation_desc_.raw ()->getInvalidator ();
    if (i)
        i->deprecateCache(Signal::Interval::Interval_ALL);
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
        if (block_update_queue)
            block_update_queue->clear();

    tfr_map_.write ()->amplitude_axis( x );

    Signal::Processing::IInvalidator::ptr i =
            render_operation_desc_.raw ()->getInvalidator ();
    if (i)
        i->deprecateCache(Signal::Interval::Interval_ALL);
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
        o.unlock ();

        Signal::Processing::IInvalidator::ptr i =
                render_operation_desc_.raw ()->getInvalidator ();
        if (i)
            i->deprecateCache(Signal::Interval::Interval_ALL);
    }
}


Signal::OperationDesc::Extent RenderModel::
        recompute_extent()
{
    if (!chain_)
        return Signal::OperationDesc::Extent();

    Signal::OperationDesc::Extent extent = chain_.read ()->extent(target_marker_);
    extent.interval           = extent.interval          .get_value_or (Signal::Interval());
    extent.number_of_channels = extent.number_of_channels.get_value_or (1);
    extent.sample_rate        = extent.sample_rate       .get_value_or (1);
    set_extent (extent);
    return extent;
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


void RenderModel::
        setPosition( Heightmap::Position pos )
{
    float l = tfr_mapping()->length();
    float x = pos.time;
    if (x<0) x=0;
    if (x>l) x=l;

    float z = pos.scale;
    if (z<0) z=0;

    if (z>1) z=1;

    camera.q[0] = x;
    camera.q[2] = z;
}


Heightmap::Position RenderModel::
        position() const
{
    return Heightmap::Position(camera.q[0], camera.q[2]);
}

} // namespace Tools
