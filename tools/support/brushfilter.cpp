#include "brushfilter.h"
#include "brushfilter.cu.h"

#include <boost/foreach.hpp>

namespace Tools {
namespace Support {

BrushFilter::
        BrushFilter()
{
    images.reset( new BrushImages );
}


void BrushFilter::
        validateRefs(Heightmap::Collection* collection)
{
    if (images->size())
    {
        if (images->begin()->first.collection() != collection)
        {
            // This happens for the first getImage after deserialization
            BrushImagesP newImages(new BrushImages);
            foreach(BrushImages::value_type bv, *images)
            {
                Heightmap::Reference rcopy( collection );
                rcopy.log2_samples_size = bv.first.log2_samples_size;
                rcopy.block_index = bv.first.block_index;
                (*newImages)[ rcopy ] = bv.second;
            }
            images = newImages;
        }
    }
}


BrushFilter::BrushImageDataP BrushFilter::
        getImage(Heightmap::Reference const& ref)
{
	validateRefs(ref.collection());
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


Signal::Intervals MultiplyBrush::
        affected_samples()
{
    Signal::Intervals r;

    BrushImages const& imgs = *images.get();



    BOOST_FOREACH(BrushImages::value_type const& v, imgs)
    {
        r |= v.first.getInterval();
    }

    return r;
}

void MultiplyBrush::
        operator()( Tfr::Chunk& chunk )
{
    BrushImages const& imgs = *images.get();

    /*float2 *f = chunk.transform_data->getCpuMemory();
    for (unsigned i=0; i <chunk.transform_data->getNumberOfElements1D(); ++i)
        f[i] = make_float2(1,0);*/
    //cudaMemset( chunk.transform_data->getCudaGlobal().ptr(), 0,
    //            chunk.transform_data->getSizeInBytes1D() );

    BOOST_FOREACH(BrushImages::value_type const& v, imgs)
    {
        Heightmap::Position a, b;
        v.first.getArea(a, b);

        Tfr::FreqAxis const& heightmapAxis = v.first.collection()->display_scale();
        float scale1 = heightmapAxis.getFrequencyScalar( chunk.min_hz );
        float scale2 = heightmapAxis.getFrequencyScalar( chunk.max_hz );

        ::multiply(
                make_float4(chunk.chunk_offset/chunk.sample_rate, scale1,
                            (chunk.chunk_offset + chunk.nSamples())/chunk.sample_rate, scale2),
                chunk.transform_data->getCudaGlobal(),
                make_float4(a.time, a.scale, b.time, b.scale),
                v.second->getCudaGlobal());
    }
}


} // namespace Support
} // namespace Tools
