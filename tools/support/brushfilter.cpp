#include "brushfilter.h"
#include "brushfiltersupport.h"

#include "brushfilter.cu.h"
#include "tfr/cwt.h"

#include <CudaException.h>


namespace Tools {
namespace Support {


BrushFilter::
        BrushFilter()
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
        img.reset( new GpuCpuData<float>(
            0,
            make_cudaExtent( ref.samplesPerBlock(), ref.scalesPerBlock(), 1),
            GpuCpuVoidData::CudaGlobal ) );
        cudaMemset( img->getCudaGlobal().ptr(), 0, img->getSizeInBytes1D() );
    }

    return img;
}


void BrushFilter::
        release_extra_resources()
{
    BrushImages const& imgs = *images.get();

    foreach(BrushImages::value_type const& v, imgs)
    {
        v.second->getCpuMemory();
        v.second->freeUnused();
    }
}


Signal::Intervals MultiplyBrush::
        affected_samples()
{
//    return Signal::Interval(0, number_of_samples());
    Signal::Intervals r;

    BrushImages const& imgs = *images.get();

    foreach(BrushImages::value_type const& v, imgs)
    {
        r |= v.first.getInterval();
    }

    return include_time_support(r);
}


void MultiplyBrush::
        operator()( Tfr::Chunk& chunk )
{
    CudaException_ThreadSynchronize();

    BrushImages const& imgs = *images.get();

    if (imgs.empty())
        return;

    Tfr::FreqAxis const& heightmapAxis = imgs.begin()->first.collection()->display_scale();
    float scale1 = heightmapAxis.getFrequencyScalar( chunk.min_hz );
    float scale2 = heightmapAxis.getFrequencyScalar( chunk.max_hz );
    float time1 = chunk.chunk_offset/chunk.sample_rate;
    float time2 = time1 + (chunk.nSamples()-1)/chunk.sample_rate;

    foreach (BrushImages::value_type const& v, imgs)
    {
        Heightmap::Position a, b;
        v.first.getArea(a, b);

        ::multiply(
                make_float4(time1, scale1,
                            time2, scale2),
                chunk.transform_data->getCudaGlobal(),
                make_float4(a.time, a.scale, b.time, b.scale),
                v.second->getCudaGlobal());
        v.second->freeUnused();
    }

    CudaException_ThreadSynchronize();
}


} // namespace Support
} // namespace Tools
