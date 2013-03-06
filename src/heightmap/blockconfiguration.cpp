#include "blockconfiguration.h"

#include "collection.h"
#include "tfr/transform.h"
#include "signal/operation.h"

namespace Heightmap {

BlockSize::
        BlockSize(int texels_per_row, int texels_per_column)
    :
        texels_per_column_( texels_per_column ),
        texels_per_row_( texels_per_row )
{
    EXCEPTION_ASSERT_LESS( 1, texels_per_row );
    EXCEPTION_ASSERT_LESS( 1, texels_per_column );
}


int BlockSize::
        texels_per_row() const
{
    return texels_per_row_;
}


int BlockSize::
        texels_per_column() const
{
    return texels_per_column_;
}


BlockConfiguration::
        BlockConfiguration( BlockSize block_size, float fs )
    :
      block_size_( block_size ),
      sample_rate_( fs ),
      amplitude_axis_(AmplitudeAxis_5thRoot)
{
    EXCEPTION_ASSERT_LESS( 0, fs );
    display_scale_.setLinear( fs );
}


int BlockConfiguration::
        samplesPerBlock() const
{
    return block_size_.texels_per_row ();
}


int BlockConfiguration::
        scalesPerBlock() const
{
    return block_size_.texels_per_column ();
}


BlockSize BlockConfiguration::
    block_size() const
{
    return block_size_;
}


Tfr::FreqAxis BlockConfiguration::
        display_scale() const
{
    return display_scale_;
}


AmplitudeAxis BlockConfiguration::
        amplitude_axis() const
{
    return amplitude_axis_;
}


void BlockConfiguration::
        display_scale(Tfr::FreqAxis v)
{
    display_scale_ = v;
}


void BlockConfiguration::
        amplitude_axis(AmplitudeAxis v)
{
    amplitude_axis_ = v;
}


float BlockConfiguration::
        targetSampleRate() const
{
    return sample_rate_;
}


} // namespace Heightmap
