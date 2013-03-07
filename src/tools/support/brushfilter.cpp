#include "brushfilter.h"
#include "brushfiltersupport.h"

#include "brushfilterkernel.h"
#include "tfr/chunk.h"
#include "heightmap/referenceinfo.h"

#include "cpumemorystorage.h"

namespace Tools {
namespace Support {


BrushFilter::
        BrushFilter(Heightmap::BlockConfiguration block_configuration)
    :
      block_configuration_(block_configuration)
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
        Heightmap::BlockSize block_size = block_configuration_.block_size ();
        img.reset( new DataStorage<float>( block_size.texels_per_row (), block_size.texels_per_column (), 1));
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
        MultiplyBrush(Heightmap::BlockConfiguration block_configuration)
    :
      BrushFilter(block_configuration)
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
        r |= v.first.getInterval();
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


bool MultiplyBrush::
        operator()( Tfr::Chunk& chunk )
{
    BrushImages const& imgs = *images.get();

    if (imgs.empty())
        return false;

    float scale1 = this->block_configuration_.display_scale ().getFrequencyScalar( chunk.minHz() );
    float scale2 = this->block_configuration_.display_scale ().getFrequencyScalar( chunk.maxHz() );
    float time1 = (chunk.chunk_offset/chunk.sample_rate).asFloat();
    float time2 = time1 + (chunk.nSamples()-1)/chunk.sample_rate;

    ResampleArea cwtArea( time1, scale1, time2, scale2 );
    foreach (BrushImages::value_type const& v, imgs)
    {
        Heightmap::Region r = Heightmap::ReferenceRegion( block_configuration_)( v.first );

        ResampleArea imgarea( r.a.time, r.a.scale, r.b.time, r.b.scale );

        ::multiply(
                cwtArea,
                chunk.transform_data,
                imgarea,
                v.second);
    }

    return true;
}


} // namespace Support
} // namespace Tools
