#include "brushfilter.h"
#include "brushfilter.cu.h"

namespace Tools {
namespace Support {


void MultiplyBrush::
        operator()( Tfr::Chunk& chunk )
{
    BrushImages const& imgs = *images.get();

    for (unsigned i=0; i<imgs.size(); ++i)
    {
        BrushImage const& img = imgs[i];
        float scaley1 = chunk.freqAxis().getFrequencyScalar( img.min_hz );
        float scaley2 = chunk.freqAxis().getFrequencyScalar( img.max_hz );

        multiply(
                make_float4(chunk.startTime(), 0, chunk.endTime(), 1),
                chunk.transform_data->getCudaGlobal(),
                make_float4(img.startTime, scaley1, img.endTime, scaley2),
                img.data->getCudaGlobal());
    }
}


} // namespace Support
} // namespace Tools
