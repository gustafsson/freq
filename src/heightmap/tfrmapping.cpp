#include "tfrmapping.h"
#include "exceptionassert.h"

namespace Heightmap {


TfrMapping::
        TfrMapping( BlockSize block_size, float fs )
    :
      block_size_( block_size ),
      sample_rate_( fs ),
      amplitude_axis_(AmplitudeAxis_5thRoot)
{
    EXCEPTION_ASSERT_LESS( 0, fs );
    display_scale_.setLinear( fs );
}


BlockSize TfrMapping::
    block_size() const
{
    return block_size_;
}


Tfr::FreqAxis TfrMapping::
        display_scale() const
{
    return display_scale_;
}


AmplitudeAxis TfrMapping::
        amplitude_axis() const
{
    return amplitude_axis_;
}


void TfrMapping::
        display_scale(Tfr::FreqAxis v)
{
    display_scale_ = v;
}


void TfrMapping::
        amplitude_axis(AmplitudeAxis v)
{
    amplitude_axis_ = v;
}


float TfrMapping::
        targetSampleRate() const
{
    return sample_rate_;
}

} // namespace Heightmap
