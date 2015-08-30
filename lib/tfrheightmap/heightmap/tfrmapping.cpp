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


float TransformDetailInfo::
        displayedFrequencyResolution( float FS, float hz1, float hz2 ) const
{
    Tfr::FreqAxis tfa = p_->freqAxis (FS);
    float scalara = tfa.getFrequencyScalar(hz1);
    float scalara2 = tfa.getFrequencyScalar(hz2);

    return scalara2 - scalara;
}


TfrMapping::
        TfrMapping( BlockLayout block_layout, int channels )
    :
      block_layout_(block_layout),
      visualization_params_(new VisualizationParams),
      length_samples_( 0 )
{
    LOGINFO TaskInfo ti("TfrMapping. Fs=%g. %d x %d blocks. mipmaps=%d. %d channels",
                block_layout_.targetSampleRate (),
                block_layout_.texels_per_row(),
                block_layout_.texels_per_column (),
                block_layout_.mipmaps (),
                channels);

    if (!Heightmap::Render::BlockTextures::isInitialized ())
        Heightmap::Render::BlockTextures::init (block_layout.texels_per_row (), block_layout.texels_per_column ()),

    this->channels (channels);
}


TfrMapping::
        ~TfrMapping()
{
    LOGINFO TaskInfo ti("~TfrMapping. Fs=%g. %d x %d blocks. mipmaps=%d. %d channels",
                block_layout_.targetSampleRate (),
                block_layout_.texels_per_row(),
                block_layout_.texels_per_column (),
                block_layout_.mipmaps (),
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

    EXCEPTION_ASSERT_EQUALS(bl.texels_per_row (), block_layout_.texels_per_row ());
    EXCEPTION_ASSERT_EQUALS(bl.texels_per_column (), block_layout_.texels_per_column ());
    // if this fails, BlockTextures would have to change size, but block_layout may then
    // be different in different windows.

    float oldfs = block_layout_.sample_rate ();
    block_layout_ = bl;
    float fs = block_layout_.sample_rate ();

    if (oldfs != fs)
    {
        Heightmap::FreqAxis fa = display_scale ();
        switch(fa.axis_scale) {
        case AxisScale_Waveform:
            fa.setWaveform (-1,1);
            break;
        case AxisScale_Linear:
            fa.setLinear (fs);
            break;
        case AxisScale_Logarithmic:
        {
            float q = fa.min_hz / fa.max_hz ();
            fa.setLogarithmic (q*fs/2,fs/2);
            break;
        }
        case AxisScale_Quefrency:
            fa.setQuefrency (fs, fa.max_frequency_scalar*2);
            break;
        default:
            break;
        }
        display_scale( fa );
    }

    updateCollections();
}


FreqAxis TfrMapping::
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
        display_scale(FreqAxis v)
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

    block_layout(BlockLayout(
                block_layout_.texels_per_row (),
                block_layout_.texels_per_column (),
                v, block_layout_.mipmaps ()));
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


double TfrMapping::
        length() const
{
    return length_samples_ / targetSampleRate();
}


Signal::IntervalType TfrMapping::
        lengthSamples() const
{
    return length_samples_;
}


void TfrMapping::
        lengthSamples(Signal::IntervalType L)
{
    if (L == length_samples_)
        return;

    length_samples_ = L;

    for (unsigned c=0; c<collections_.size(); ++c)
        collections_[c].write ()->length( length() );
}


int TfrMapping::
        channels() const
{
    return (int)collections_.size ();
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

    for (auto c : collections_)
        old_collections_.push_back (std::move(c));

    Collections new_collections(v);

    for (pCollection& c : new_collections)
    {
        c = Heightmap::Collection::ptr( new Heightmap::Collection(block_layout_, visualization_params_));
        c->length( length_samples_ );
    }

    collections_.swap (new_collections);
}


TfrMapping::Collections TfrMapping::
        collections() const
{
    return collections_;
}


void TfrMapping::
        gc()
{
    old_collections_.clear ();
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
#include <QtWidgets> // QApplication
#include <QtOpenGL> // QGLWidget

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
