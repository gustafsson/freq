#include "tfrmapping.h"
#include "exceptionassert.h"
#include "collection.h"

namespace Heightmap {

TfrMapping::
        TfrMapping( BlockSize block_size, float fs )
    :
      block_size( block_size ),
      targetSampleRate( fs ),
      length( 0 ),
      amplitude_axis(AmplitudeAxis_5thRoot)
{
    EXCEPTION_ASSERT_LESS( 0, fs );
    display_scale.setLinear( fs );
    // by default there is no transform_desc, and nothing will be drawn
}


bool TfrMapping::
        operator==(const TfrMapping& b)
{
    return block_size == b.block_size &&
            targetSampleRate == b.targetSampleRate &&
            length == b.length &&
            transform_desc == b.transform_desc &&
            (transform_desc ? *transform_desc == *b.transform_desc : true) &&
            display_scale == b.display_scale &&
            amplitude_axis == b.amplitude_axis;
}


TfrMap::
        TfrMap( TfrMapping tfr_mapping, int channels )
    :
      tfr_mapping_( tfr_mapping )
{
    this->channels (channels);
}


TfrMap::
        ~TfrMap()
{
    collections_.clear();
}


BlockSize TfrMap::
    block_size() const
{
    return tfr_mapping_.block_size;
}


void TfrMap::
        block_size(BlockSize bs)
{
    if (bs == tfr_mapping_.block_size)
        return;

    tfr_mapping_.block_size = bs;

    updateCollections();
}


Tfr::FreqAxis TfrMap::
        display_scale() const
{
    return tfr_mapping_.display_scale;
}


AmplitudeAxis TfrMap::
        amplitude_axis() const
{
    return tfr_mapping_.amplitude_axis;
}


void TfrMap::
        display_scale(Tfr::FreqAxis v)
{
    if (v == tfr_mapping_.display_scale)
        return;

    tfr_mapping_.display_scale = v;

    updateCollections();
}


void TfrMap::
        amplitude_axis(AmplitudeAxis v)
{
    if (v == tfr_mapping_.amplitude_axis)
        return;

    tfr_mapping_.amplitude_axis = v;

    updateCollections();
}


float TfrMap::
        targetSampleRate() const
{
    return tfr_mapping_.targetSampleRate;
}


void TfrMap::
        targetSampleRate(float v)
{
    if (v == tfr_mapping_.targetSampleRate)
        return;

    tfr_mapping_.targetSampleRate = v;

    updateCollections();
}


Tfr::TransformDesc::Ptr TfrMap::
        transform_desc() const
{
    return tfr_mapping_.transform_desc;
}


void TfrMap::
        transform_desc(Tfr::TransformDesc::Ptr t)
{
    if (t == tfr_mapping_.transform_desc)
        return;

    if (t && tfr_mapping_.transform_desc && (*t == *tfr_mapping_.transform_desc))
        return;

    tfr_mapping_.transform_desc = t;

    updateCollections();
}


const TfrMapping& TfrMap::
        tfr_mapping() const
{
    return tfr_mapping_;
}


float TfrMap::
        length() const
{
    return tfr_mapping_.length;
}


void TfrMap::
        length(float L)
{
    if (L == tfr_mapping_.length)
        return;

    tfr_mapping_.length = L;

    updateCollections();
}


int TfrMap::
        channels() const
{
    return collections_.size ();
}


void TfrMap::
        channels(int v)
{
    // There are several assumptions that there is at least one channel on the form 'collections()[0]'
    if (v < 1)
        v = 1;

    if (v == channels())
        return;

    collections_.clear ();
    collections_.resize(v);

    for (int c=0; c<v; ++c)
    {
        collections_[c].reset( new Heightmap::Collection(tfr_mapping_));
    }
}


TfrMap::Collections TfrMap::
        collections() const
{
    return collections_;
}


void TfrMap::
        updateCollections()
{
    for (unsigned c=0; c<collections_.size(); ++c)
        write1(collections_[c])->tfr_mapping( tfr_mapping_ );
}


void TfrMap::
        test()
{
    TfrMap::Ptr t = testInstance();
    write1(t)->block_size( BlockSize(123,456) );
    EXCEPTION_ASSERT_EQUALS( BlockSize(123,456), read1(t)->tfr_mapping().block_size );
}


} // namespace Heightmap

#include "tfr/stftdesc.h"

namespace Heightmap
{

TfrMap::Ptr TfrMap::
        testInstance()
{
    TfrMapping tfrmapping(BlockSize(1<<8, 1<<8), 10);
    TfrMap::Ptr tfrmap(new TfrMap(tfrmapping, 1));
    write1(tfrmap)->transform_desc( Tfr::StftDesc ().copy ());
    return tfrmap;
}

} // namespace Heightmap
