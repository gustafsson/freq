#include "tfrmapping.h"
#include "exceptionassert.h"
#include "collection.h"

namespace Heightmap {

TfrMapping::
        TfrMapping( BlockLayout block_layout, int channels )
    :
      block_layout_(block_layout),
      visualization_params_(new VisualizationParams),
      length_( 0 )
{
    this->channels (channels);
}


TfrMapping::
        ~TfrMapping()
{
    collections_.clear();
}


BlockLayout TfrMapping::
    block_layout() const
{
    return block_layout_;
}


void TfrMapping::
        block_layout(BlockLayout bl)
{
    if (bl == block_layout_)
        return;

    block_layout_ = bl;

    updateCollections();
}


Tfr::FreqAxis TfrMapping::
        display_scale() const
{
    return visualization_params_->display_scale();
}


AmplitudeAxis TfrMapping::
        amplitude_axis() const
{
    return visualization_params_->amplitude_axis();
}


void TfrMapping::
        display_scale(Tfr::FreqAxis v)
{
    if (v == visualization_params_->display_scale())
        return;

    visualization_params_->display_scale( v );

    updateCollections();
}


void TfrMapping::
        amplitude_axis(AmplitudeAxis v)
{
    if (v == visualization_params_->amplitude_axis())
        return;

    visualization_params_->amplitude_axis( v );

    updateCollections();
}


float TfrMapping::
        targetSampleRate() const
{
    return block_layout_.targetSampleRate ();
}


void TfrMapping::
        targetSampleRate(float v)
{
    if (v == block_layout_.targetSampleRate ())
        return;

    block_layout_ = BlockLayout(
                block_layout_.texels_per_row (),
                block_layout_.texels_per_column (),
                v);

    updateCollections();
}


Tfr::TransformDesc::Ptr TfrMapping::
        transform_desc() const
{
    return visualization_params_->transform_desc();
}


void TfrMapping::
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


float TfrMapping::
        length() const
{
    return length_;
}


void TfrMapping::
        length(float L)
{
    if (L == length_)
        return;

    length_ = L;

    for (unsigned c=0; c<collections_.size(); ++c)
        write1(collections_[c])->length( length_ );
}


int TfrMapping::
        channels() const
{
    return collections_.size ();
}


void TfrMapping::
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
        write1(collections_[c])->length( length_ );
    }
}


TfrMapping::Collections TfrMapping::
        collections() const
{
    return collections_;
}


void TfrMapping::
        updateCollections()
{
    for (unsigned c=0; c<collections_.size(); ++c)
        write1(collections_[c])->block_layout( block_layout_ );

    for (unsigned c=0; c<collections_.size(); ++c)
        write1(collections_[c])->visualization_params( visualization_params_ );
}

} // namespace Heightmap


#include "tfr/stftdesc.h"
#include <QApplication>
#include <QGLWidget>

namespace Heightmap
{

void TfrMapping::
        test()
{
    std::string name = "TfrMapping";
    int argc = 1;
    char * argv = &name[0];
    QApplication a(argc,&argv);
    QGLWidget w;
    w.makeCurrent ();

    {
        TfrMapping::Ptr t = testInstance();
        write1(t)->block_layout( BlockLayout(123,456,789) );
        EXCEPTION_ASSERT_EQUALS( BlockLayout(123,456,789), read1(t)->block_layout() );
    }
}


TfrMapping::Ptr TfrMapping::
        testInstance()
{
    BlockLayout bl(1<<8, 1<<8, 10);
    TfrMapping::Ptr tfrmap(new TfrMapping(bl, 1));
    write1(tfrmap)->transform_desc( Tfr::StftDesc ().copy ());
    return tfrmap;
}

} // namespace Heightmap
