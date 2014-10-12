#include "renderviewtransform.h"

#include "heightmap/tfrmappings/stftblockfilter.h"
#include "heightmap/tfrmappings/waveformblockfilter.h"
#include "heightmap/tfrmapping.h"
#include "heightmap/update/updateproducer.h"
#include "tfr/stftdesc.h"
#include "tfr/waveformrepresentation.h"
#include "tfr/transformoperation.h"
#include "signal/operationwrapper.h"
#include "timer.h"
#include "log.h"
#include "demangle.h"

RenderViewTransform::RenderViewTransform(Tools::RenderModel& render_model)
    :
      render_model(render_model)
{

}


void RenderViewTransform::
        receiveSetTransform_Stft()
{
    Tfr::StftDesc& stft = render_model.transform_descs()->getParam<Tfr::StftDesc>();
    stft.setWindow (Tfr::StftDesc::WindowType_Hann, 1 - 1/4.);
    stft.set_approximate_chunk_size (1<<12);
    stft.enable_inverse (false);

    // Setup the kernel that will take the transform data and create an image
    Heightmap::TfrMappings::StftBlockFilterParams::ptr sbfp{new Heightmap::TfrMappings::StftBlockFilterParams};
    Heightmap::MergeChunkDesc::ptr mcdp(new Heightmap::TfrMappings::StftBlockFilterDesc(sbfp));

    // Get a copy of the transform to use
    setBlockFilter(mcdp, stft.copy());
}


void RenderViewTransform::
        receiveSetTransform_Waveform()
{
    Tfr::WaveformRepresentationDesc& dw = render_model.transform_descs()->getParam<Tfr::WaveformRepresentationDesc>();

    // Setup the kernel that will take the transform data and create an image
    Heightmap::MergeChunkDesc::ptr mcdp(new Heightmap::TfrMappings::WaveformBlockFilterDesc);

    // Get a copy of the transform to use
    setBlockFilter(mcdp, dw.copy());
}


void RenderViewTransform::
        setBlockFilter(Heightmap::MergeChunkDesc::ptr mcdp, Tfr::TransformDesc::ptr transform_desc)
{
    // Wire it up to a FilterDesc
    Heightmap::Update::UpdateProducerDesc* cbfd;
    Tfr::ChunkFilterDesc::ptr kernel(cbfd
            = new Heightmap::Update::UpdateProducerDesc(render_model.update_queue(), render_model.tfr_mapping ()));
    cbfd->setMergeChunkDesc( mcdp );
    kernel.write ()->transformDesc(transform_desc);
    setBlockFilter( kernel );
}


void RenderViewTransform::
        setBlockFilter(Tfr::ChunkFilterDesc::ptr kernel)
{
    Tfr::TransformOperationDesc::ptr adapter( new Tfr::TransformOperationDesc(kernel));
    // Ambiguity
    // Tfr::TransformOperationDesc defines a current transformDesc
    // VisualizationParams also defines a current transformDesc

    render_model.set_filter (adapter);

    // abort target needs
    auto step = render_model.target_marker ()->target_needs ()->step ().lock (); // lock weak_ptr
    render_model.target_marker ()->target_needs ()->updateNeeds (Signal::Intervals());
    int sleep_ms = 1000;
    Timer t;
    for (int i=0; i<sleep_ms && !Signal::Processing::Step::sleepWhileTasks (step.read(), 1); i++)
    {
        render_model.update_queue()->clear ();
    }

    if (!Signal::Processing::Step::sleepWhileTasks (step.read(), 1))
        Log("RenderViewTransform: didn't finish in %g ms, changing anyway to %s") % t.elapsed () % vartype(*kernel.raw ()->transformDesc ());

    // then change the tfr_mapping
    render_model.tfr_mapping ()->transform_desc( kernel.raw ()->transformDesc ()->copy() );
}
