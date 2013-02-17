#include "blockconfiguration.h"

#include "collection.h"
#include "tfr/transform.h"
#include "signal/operation.h"

namespace Heightmap {


BlockConfiguration::
        BlockConfiguration( Collection* collection )
    :   collection_(collection)
{}


Collection* BlockConfiguration::
        collection() const
{
    return this->collection_;
}


void BlockConfiguration::
        setCollection(Collection* c)
{
    this->collection_ = c;
}


unsigned BlockConfiguration::
        samplesPerBlock() const
{
    return this->collection_->samples_per_block ();
}


unsigned BlockConfiguration::
        scalesPerBlock() const
{
    return this->collection_->scales_per_block ();
}


float BlockConfiguration::
        targetSampleRate() const
{
    return this->collection_->target->sample_rate ();
}


Tfr::FreqAxis BlockConfiguration::
        display_scale() const
{
    return this->collection_->display_scale ();
}


Tfr::FreqAxis BlockConfiguration::
        transform_scale() const
{
    return collection_->transform()->freqAxis(targetSampleRate ());
}


float BlockConfiguration::
        displayedTimeResolution(float hz) const
{
    return collection_->transform()->displayedTimeResolution(targetSampleRate (), hz);
}


float BlockConfiguration::
        length() const
{
    return collection_->target->length();
}


} // namespace Heightmap
