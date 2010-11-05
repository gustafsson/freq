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


void MultiplyBrush::
        operator()( Tfr::Chunk& chunk )
{
    BrushImages const& imgs = *images.get();

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
