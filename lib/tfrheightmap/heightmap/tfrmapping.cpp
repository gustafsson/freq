#include "tfrmapping.h"

#include "heightmap/collection.h"

#include "exceptionassert.h"
#include "tasktimer.h"

//#define LOGINFO
#define LOGINFO if(0)

namespace Heightmap {

TransformDetailInfo::
        TransformDetailInfo(Tfr::TransformDesc::ptr p)
    :
      p_(p)
{

}


bool TransformDetailInfo::
        operator==(const DetailInfo& db) const
{
    const TransformDetailInfo* b = dynamic_cast<const TransformDetailInfo*>(&db);
    if (!b)
        return false;
    if (p_ && b->p_)
        return *p_ == *b->p_;
    else
        return p_ == b->p_;
}


float TransformDetailInfo::
        displayedTimeResolution( float FS, float hz ) const
{
    return p_->displayedTimeResolution (FS, hz);
}


Tfr::FreqAxis TransformDetailInfo::
        freqAxis( float fs ) const
{
    return p_->freqAxis (fs);
}


TfrMapping::
        TfrMapping( BlockLayout block_layout, int channels )
    :
      block_layout_(block_layout),
      visualization_params_(new VisualizationParams),
      length_( 0 )
{
    LOGINFO TaskInfo ti("TfrMapping. Fs=%g. %d x %d blocks",
                block_layout_.targetSampleRate (),
                block_layout_.texels_per_row(),
                block_layout_.texels_per_column ());

    this->channels (channels);
}


TfrMapping::
        ~TfrMapping()
{
    LOGINFO TaskInfo ti("~TfrMapping. Fs=%g. %d x %d blocks. %d channels",
                block_layout_.targetSampleRate (),
                block_layout_.texels_per_row(),
                block_layout_.texels_per_column (),
                channels ());

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

    LOGINFO TaskInfo ti("Target sample rate: %g. %d x %d blocks",
                bl.targetSampleRate (),
                bl.texels_per_row(),
                bl.texels_per_column ());

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

    LOGINFO TaskInfo ti("Target sample rate: %g", v);

    block_layout_ = BlockLayout(
                block_layout_.texels_per_row (),
                block_layout_.texels_per_column (),
                v);

    updateCollections();
}


Tfr::TransformDesc::ptr TfrMapping::
        transform_desc() const
{
    DetailInfo::ptr d = visualization_params_->detail_info ();
    TransformDetailInfo* t = dynamic_cast<TransformDetailInfo*>(d.get ());
    return t ? t->transform_desc() : Tfr::TransformDesc::ptr();
}


void TfrMapping::
        transform_desc(Tfr::TransformDesc::ptr t)
{
    Tfr::TransformDesc::ptr o = transform_desc();
    if (t == o)
        return;

    if (t && o && (*t == *o))
        return;

    DetailInfo::ptr d(new TransformDetailInfo(t));
    visualization_params_->detail_info (d);

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
        collections_[c].write ()->length( length_ );
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

    LOGINFO TaskInfo ti("Number of channels: %d", v);

    collections_.clear ();

    Collections new_collections(v);

    for (pCollection& c : new_collections)
    {
        c = Heightmap::Collection::ptr( new Heightmap::Collection(block_layout_, visualization_params_));
        c->length( length_ );
    }

    collections_ = new_collections;
}


TfrMapping::Collections TfrMapping::
        collections() const
{
    return collections_;
}


void TfrMapping::
        updateCollections()
{
    for (pCollection c : collections_)
        c->block_layout( block_layout_ );

    for (pCollection c : collections_)
        c->visualization_params( visualization_params_ );
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
        TfrMapping::ptr t = testInstance();
        t.write ()->block_layout( BlockLayout(123,456,789) );
        EXCEPTION_ASSERT_EQUALS( BlockLayout(123,456,789), t.read ()->block_layout() );
    }
}


TfrMapping::ptr TfrMapping::
        testInstance()
{
    BlockLayout bl(1<<8, 1<<8, 10);
    TfrMapping::ptr tfrmap(new TfrMapping(bl, 1));
    tfrmap.write ()->transform_desc( Tfr::StftDesc ().copy ());
    return tfrmap;
}

} // namespace Heightmap
