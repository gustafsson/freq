#include "blockconfiguration.h"

#include "collection.h"
#include "tfr/transform.h"
#include "signal/operation.h"

namespace Heightmap {


BlockConfiguration::
        BlockConfiguration( Collection* collection )
    :
      collection_(collection),
      scales_per_block_( -1 ),
      samples_per_block_( -1 ),
      amplitude_axis_(AmplitudeAxis_5thRoot)
{
    display_scale_.setLinear(1);
}


unsigned BlockConfiguration::
        samplesPerBlock() const
{
    return samples_per_block_;
}


unsigned BlockConfiguration::
        scalesPerBlock() const
{
    return scales_per_block_;
}


void BlockConfiguration::
        samplesPerBlock(unsigned spb)
{
    samples_per_block_ = spb;
}


void BlockConfiguration::
        scalesPerBlock(unsigned spb)
{
    scales_per_block_ = spb;
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
    return collection_->target->sample_rate ();
}


float BlockConfiguration::
        length() const
{
    return collection_->target->length();
}


} // namespace Heightmap
