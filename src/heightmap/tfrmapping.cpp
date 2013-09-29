#include "tfrmapping.h"
#include "exceptionassert.h"
#include "collection.h"

namespace Heightmap {

TfrMap::
        TfrMap( BlockLayout block_layout, int channels )
    :
      block_layout_(block_layout),
      visualization_params_(new VisualizationParams),
      length_( 0 )
{
    this->channels (channels);
}


TfrMap::
        ~TfrMap()
{
    collections_.clear();
}


BlockLayout TfrMap::
    block_layout() const
{
    return block_layout_;
}


void TfrMap::
        block_layout(BlockLayout bl)
{
    if (bl == block_layout_)
        return;

    block_layout_ = bl;

    updateCollections();
}


Tfr::FreqAxis TfrMap::
        display_scale() const
{
    return visualization_params_->display_scale();
}


AmplitudeAxis TfrMap::
        amplitude_axis() const
{
    return visualization_params_->amplitude_axis();
}


void TfrMap::
        display_scale(Tfr::FreqAxis v)
{
    if (v == visualization_params_->display_scale())
        return;

    visualization_params_->display_scale( v );

    updateCollections();
}


void TfrMap::
        amplitude_axis(AmplitudeAxis v)
{
    if (v == visualization_params_->amplitude_axis())
        return;

    visualization_params_->amplitude_axis( v );

    updateCollections();
}


float TfrMap::
        targetSampleRate() const
{
    return block_layout_.targetSampleRate ();
}


void TfrMap::
        targetSampleRate(float v)
{
    if (v == block_layout_.targetSampleRate ())
        return;

    block_layout_ = BlockLayout(
                block_layout_.texels_per_row (),
                block_layout_.texels_per_column (),
                v);

    Tfr::FreqAxis f;
    f.setLinear( v );
    visualization_params_->display_scale(f);

    updateCollections();
}


Tfr::TransformDesc::Ptr TfrMap::
        transform_desc() const
{
    return visualization_params_->transform_desc();
}


void TfrMap::
        transform_desc(Tfr::TransformDesc::Ptr t)
{
    VisualizationParams::Ptr vp = visualization_params_;
    if (t == vp->transform_desc())
        return;

    if (t && vp->transform_desc() && (*t == *vp->transform_desc()))
        return;

    visualization_params_->transform_desc( t );

    updateCollections();
}


//const TfrMapping& TfrMap::
//        tfr_mapping() const
//{
//    return tfr_mapping_;
//}


float TfrMap::
        length() const
{
    return length_;
}


void TfrMap::
        length(float L)
{
    if (L == length_)
        return;

    length_ = L;

    for (unsigned c=0; c<collections_.size(); ++c)
        write1(collections_[c])->length( length_ );
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
        collections_[c].reset( new Heightmap::Collection(block_layout_, visualization_params_));
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
        write1(collections_[c])->block_layout( block_layout_ );

    for (unsigned c=0; c<collections_.size(); ++c)
        write1(collections_[c])->visualization_params( visualization_params_ );
}


void TfrMap::
        test()
{
    TfrMap::Ptr t = testInstance();
    write1(t)->block_layout( BlockLayout(123,456,789) );
    EXCEPTION_ASSERT_EQUALS( BlockLayout(123,456,789), read1(t)->block_layout() );
}


} // namespace Heightmap

#include "tfr/stftdesc.h"

namespace Heightmap
{

TfrMap::Ptr TfrMap::
        testInstance()
{
    BlockLayout bl(1<<8, 1<<8, 10);
    TfrMap::Ptr tfrmap(new TfrMap(bl, 1));
    write1(tfrmap)->transform_desc( Tfr::StftDesc ().copy ());
    return tfrmap;
}

} // namespace Heightmap
