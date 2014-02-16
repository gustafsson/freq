#include "brushfilter.h"
#include "brushfiltersupport.h"

#include "brushfilterkernel.h"
#include "tfr/chunk.h"
#include "heightmap/referenceinfo.h"

#include "cpumemorystorage.h"
#include "tasktimer.h"

namespace Tools {
namespace Support {


BrushFilter::
        BrushFilter(Heightmap::BlockLayout block_layout, Heightmap::VisualizationParams::ConstPtr visualization_params)
    :
      block_layout_(block_layout),
      visualization_params_(visualization_params)
{
    images.reset( new BrushImages );
    //transform( Tfr::pTransform(new Tfr::Cwt( Tfr::Cwt::Singleton())));
    resource_releaser_ = new BrushFilterSupport(this);
}


BrushFilter::
        ~BrushFilter()
{
    TaskInfo ti("%s", __FUNCTION__);
    delete resource_releaser_;
}


//void BrushFilter::
//        validateRefs(Heightmap::Collection* collection)
//{
//    if (images->size())
//    {
//        if (images->begin()->first.collection() != collection)
//        {
//            // This happens for the first getImage after deserialization
//            BrushImagesP newImages(new BrushImages);
//            foreach(BrushImages::value_type bv, *images)
//            {
//                Heightmap::Reference rcopy( collection );
//                rcopy.log2_samples_size = bv.first.log2_samples_size;
//                rcopy.block_index = bv.first.block_index;
//                (*newImages)[ rcopy ] = bv.second;
//            }
//            images = newImages;
//        }
//    }
//}


BrushFilter::BrushImageDataP BrushFilter::
        getImage(Heightmap::Reference const& ref)
{
    //validateRefs(ref.collection());
    BrushImageDataP& img = (*images)[ ref ];

    if (!img)
    {
        img.reset( new DataStorage<float>( block_layout_.texels_per_row (), block_layout_.texels_per_column (), 1));
    }

    return img;
}


void BrushFilter::
        release_extra_resources()
{
    BrushImages const& imgs = *images.get();

    foreach(BrushImages::value_type const& v, imgs)
    {
        v.second->OnlyKeepOneStorage<CpuMemoryStorage>();
    }
}


MultiplyBrush::
        MultiplyBrush(Heightmap::BlockLayout bl, Heightmap::VisualizationParams::ConstPtr vp)
    :
      BrushFilter(bl, vp)
{

}


Signal::Intervals MultiplyBrush::
        affected_samples()
{
//    return getInterval();
    Signal::Intervals r;

    BrushImages const& imgs = *images.get();

    foreach(BrushImages::value_type const& v, imgs)
    {
        r |= Heightmap::ReferenceInfo(v.first, block_layout_, visualization_params_).getInterval();
    }

    return r;
}


std::string MultiplyBrush::
        name()
{
    std::stringstream ss;
    ss << "Brush stroke - multiplicative";
    if (images)
    {
        ss << " - " << images->size() << " block";
        if (images->size() != 1)
            ss << "s";
    }
    return ss.str();
}


void MultiplyBrush::
        operator()( Tfr::ChunkAndInverse& chunkai )
{
    Tfr::Chunk& chunk = *chunkai.chunk;
    BrushImages const& imgs = *images.get();

    if (imgs.empty()) {
        // Return dummy inverse
        // return false;
        return;
    }

    float scale1 = visualization_params_->display_scale().getFrequencyScalar( chunk.minHz() );
    float scale2 = visualization_params_->display_scale().getFrequencyScalar( chunk.maxHz() );
    float time1 = (chunk.chunk_offset/chunk.sample_rate).asFloat();
    float time2 = time1 + (chunk.nSamples()-1)/chunk.sample_rate;

    ResampleArea cwtArea( time1, scale1, time2, scale2 );
    foreach (BrushImages::value_type const& v, imgs)
    {
        Heightmap::Region r = Heightmap::RegionFactory( block_layout_ )( v.first );

        ResampleArea imgarea( r.a.time, r.a.scale, r.b.time, r.b.scale );

        ::multiply(
                cwtArea,
                chunk.transform_data,
                imgarea,
                v.second);
    }
}


MultiplyBrushDesc::
        MultiplyBrushDesc(Heightmap::BlockLayout bl, Heightmap::VisualizationParams::ConstPtr vp)
    :
      bl(bl),
      vp(vp)
{}


Tfr::pChunkFilter MultiplyBrushDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    return Tfr::pChunkFilter(new MultiplyBrush(bl, vp));
}


} // namespace Support
} // namespace Tools
