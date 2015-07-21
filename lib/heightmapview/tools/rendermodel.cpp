#include "rendermodel.h"

#include "heightmap/collection.h"
#include "heightmap/render/renderer.h"
#include "signal/operationwrapper.h"

#include "tfr/chunkfilter.h"

#include "support/renderoperation.h"

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
        camera(new Tools::Support::RenderCamera),
        gl_projection(new glProjection),
        transform_descs_(new Support::TransformDescs),
        stft_block_filter_params_(new Heightmap::TfrMappings::StftBlockFilterParams)
{
    Heightmap::BlockLayout bl(1<<8,1<<8,1,Heightmap::Render::BlockTextures::mipmaps);
    tfr_map_.reset (new Heightmap::TfrMapping(bl, 0));

    resetCameraSettings();
//    setTestCamera();
}


RenderModel::
        ~RenderModel()
{
}


void RenderModel::
        init(Signal::Processing::Chain::ptr chain,
             Heightmap::Update::UpdateQueue::ptr update_queue,
             Support::RenderOperationDesc::RenderTarget::ptr rt,
             Signal::Processing::TargetMarker::ptr target_marker)
{
    // specify wrapped filter with set_filter
    Support::RenderOperationDesc*rod;
    render_operation_desc_.reset(rod=new Support::RenderOperationDesc(Signal::OperationDesc::ptr(), rt));

    if (target_marker)
        target_marker_ = chain->addTargetAfter(render_operation_desc_,target_marker);
    else
        target_marker_ = chain->addTarget(render_operation_desc_);

    rod->setInvalidator(Signal::Processing::IInvalidator::ptr(
            new TargetInvalidator(target_marker_->target_needs ())));

    chain_ = chain;
    update_queue_ = update_queue;

    recompute_extent ();
}


void RenderModel::
        resetCameraSettings()
{
    auto camera = this->camera.write ();
    camera->q = vectord(0,0,.5f);
    camera->p = vectord(0,0,-10.f);
    camera->r = vectord(91,180,0);
    camera->xscale = -camera->p[2]*0.1f;
    camera->zscale = -camera->p[2]*0.75f;

    float L = tfr_mapping ().read ()->length();
    if (L)
    {
        camera->xscale = 10/L;
        camera->q[0] = 0.5*L;
    }

#ifdef TARGET_hast
    camera->p[2] = -6;
    xscale = 0.1f;

    if (L)
        camera->xscale = 14/L;
#endif
}


void RenderModel::
        resetBlockCaches()
{
    for (auto c : collections())
        Heightmap::Collection::cache (c)->clear();
}


void RenderModel::
        setTestCamera()
{
    auto c = camera.write();
    render_settings.y_scale = 0.01f;
    c->q = vectord(63.4565,0,0.37);
    c->p = vectord(0,0,-10);
    c->r = vectord(46.2, 253.186, 0);

    c->orthoview.reset( c->r[0] >= 90 );
}


void RenderModel::
        deprecateCache(Signal::Intervals I)
{
    if (render_operation_desc_)
        if (auto i = render_operation_desc_.raw ()->getInvalidator ())
            i->deprecateCache(I);
}


Heightmap::TfrMapping::Collections RenderModel::
        collections() const
{
    return tfr_map_.read ()->collections();
}


Signal::Processing::Chain::ptr RenderModel::
        chain()
{
    return chain_;
}


Heightmap::Update::UpdateQueue::ptr RenderModel::
        update_queue()
{
    return update_queue_;
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
        if (update_queue_)
            update_queue_->clear();

    tfr_map_.write ()->display_scale( x );

    deprecateCache();
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
        if (update_queue_)
            update_queue_->clear();

    tfr_map_.write ()->amplitude_axis( x );

    deprecateCache ();
}


Heightmap::TfrMapping::ptr RenderModel::
        tfr_mapping() const
{
    return tfr_map_;
}


Support::TransformDescs::ptr RenderModel::
        transform_descs() const
{
    return transform_descs_;
}


Tfr::TransformDesc::ptr RenderModel::
        transform_desc() const
{
    auto r = render_operation_desc_;
    if (!r) return Tfr::TransformDesc::ptr();

    auto o = r.read ();
    const Support::RenderOperationDesc* rod = dynamic_cast<const Support::RenderOperationDesc*>(&*o);

    return rod
            ? rod->transform_desc ()
            : Tfr::TransformDesc::ptr();
}


void RenderModel::
        set_transform_desc(Tfr::TransformDesc::ptr t)
{
    {
        EXCEPTION_ASSERT(render_operation_desc_);

        auto o = render_operation_desc_.write ();
        Support::RenderOperationDesc* rod = dynamic_cast<Support::RenderOperationDesc*>(&*o);

        if (!rod)
            return;

        rod->transform_desc (t);
        o.unlock ();

        // and change the tfr_mapping
        t = t->copy ();
        tfr_map_->transform_desc( t );

        deprecateCache();
    }
}


void RenderModel::
        recompute_extent()
{
    Signal::OperationDesc::Extent extent;

    if (chain_)
        extent = chain_->extent(target_marker_);

    set_extent (extent);
}


void RenderModel::
        set_extent(Signal::OperationDesc::Extent extent)
{
    auto w = tfr_map_.write ();

    //
    Signal::OperationDesc::Extent prevextent;
    prevextent.sample_rate = w->targetSampleRate();
    prevextent.number_of_channels = w->channels();
    prevextent.interval = Signal::Interval(0, w->lengthSamples());

    extent.sample_rate = extent.sample_rate.get_value_or (1);
    extent.interval = extent.interval.get_value_or (Signal::Interval());
    extent.number_of_channels = extent.number_of_channels.get_value_or (1);

    w->targetSampleRate( extent.sample_rate.get() );
    w->lengthSamples( extent.interval.get().last );
    w->channels( extent.number_of_channels.get () );
    w.unlock ();

    Signal::Intervals deprecate;
    if (prevextent.sample_rate.get () != extent.sample_rate.get () ||
            prevextent.number_of_channels.get () != prevextent.number_of_channels.get ())
    {
        deprecate = Signal::Interval::Interval_ALL;
    } else if (prevextent.interval.get () != extent.interval.get ()) {
        // should deprecate new or modified samples through OperationDesc::getInvalidator
    } else {
        // nothing changed, ok
    }

    deprecateCache(deprecate);
}


Signal::Processing::TargetMarker::ptr RenderModel::
        target_marker()
{
    return target_marker_;
}


void RenderModel::
        set_filter(Signal::OperationDesc::ptr o)
{
    EXCEPTION_ASSERT(render_operation_desc_);

    auto wo = render_operation_desc_.write ();
    Signal::OperationDescWrapper* w =
            dynamic_cast<Signal::OperationDescWrapper*>(&*wo);

    w->setWrappedOperationDesc (o);
    wo.unlock ();

    deprecateCache();
}


Signal::OperationDesc::ptr RenderModel::
        get_filter()
{
    if (!render_operation_desc_)
        return Signal::OperationDesc::ptr();

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

    auto c = camera.write ();
    c->q[0] = x;
    c->q[2] = z;
}


Heightmap::Position RenderModel::
        position() const
{
    auto c = camera.read ();
    return Heightmap::Position(c->q[0], c->q[2]);
}

} // namespace Tools
