#include "tooltipmodel.h"

#include "sawe/project.h"
#include "tfr/cwt.h"
#include "tfr/cwtfilter.h"
#include "tfr/cepstrum.h"
#include "tfr/cwtchunk.h"

#include "commentcontroller.h"
#include "heightmap/renderer.h"
#include "heightmap/collection.h"

// gpumisc
#include <TaskTimer.h>
#include <neat_math.h>

// Qt
#include <QToolTip>

// Std
#include <sstream>
#include <fstream>
using namespace std;

#ifdef min
#undef min
#endif

namespace Tools {

TooltipModel::TooltipModel()
    :
        pos_time(0),
        pos_hz(-1),
        max_so_far(-1),
        markers(0),
        comment(0),
        automarking(TooltipModel::ManualMarkers),
        comments_(0),
        render_view_(0),
        fetched_heightmap_values(0),
        last_fetched_scale_(-1)
{
}


void TooltipModel::
        setPtrs(RenderView *render_view, CommentController* comments)
{
    render_view_ = render_view;
    comments_ = comments;
    comment = comments_->findView( this->comment_model );
}


const Heightmap::Position& TooltipModel::
        comment_pos()
{
    return comment->model()->pos;
}


void TooltipModel::
        showToolTip( Heightmap::Position p )
{
    Tfr::Cwt& c = Tfr::Cwt::Singleton();

    if (TooltipModel::ManualMarkers != this->automarking)
    {
        // computeMarkerMeasure and others will set automarking back to working if it is not finished
        this->automarking = AutoMarkerFinished;
    }

    fetched_heightmap_values = 0;


    bool last_fetched_scale_is_valid = true;
    if (0<=last_fetched_scale_)
        render_view_->getHeightmapValue( p, 0, &last_fetched_scale_,
                                         false, &last_fetched_scale_is_valid );

    // Fetch local max based on quadratic interpolation (if any max exists
    // within '1 pixel') to create a more accurate position. Move 'p' to this
    // local max if any is found.
    bool is_valid_value;
    float val;
    if (last_fetched_scale_is_valid )
        val = render_view_->getHeightmapValue( p, 0, &p.scale,
                                               false, &is_valid_value );

    if (!is_valid_value || !last_fetched_scale_is_valid)
    {
        if (TooltipModel::ManualMarkers != this->automarking)
            automarking = AutoMarkerWorking;
        return;
    }

    bool found_better = val > max_so_far;
    if (found_better)
    {
        this->pos_time = p.time;
        this->pos_hz = render_view_->model->display_scale().getFrequency( p.scale );
    }
    else
        p = this->pos();

    float FS = render_view_->model->project()->worker.source()->sample_rate();
    float f = this->pos_hz;

    float best_compliance;
    if (found_better || TooltipModel::ManualMarkers != this->automarking )
        this->markers = this->markers_auto = guessHarmonicNumber( p, best_compliance );
    else
        best_compliance = computeMarkerMeasure( p, this->markers, 0);

    if (found_better)
        this->max_so_far = val;

    stringstream ss;
    ss << setiosflags(ios::fixed)
       << "Time: " << setprecision(3) << p.time << " s<br/>"
       << "Frequency: " << setprecision(1) << f << " Hz<br/>";

    if (dynamic_cast<Tfr::CwtFilter*>( render_view_->model->block_filter()))
    {
        float std_t = c.morlet_sigma_samples( FS, f ) / FS;
        float std_f = c.morlet_sigma_f( f );
        ss << "Morlet standard deviation: " << setprecision(3) << std_t << " s, " << setprecision(1) << std_f << " Hz<br/>";
    }

    ss << "Value here: " << setprecision(10) << this->max_so_far << setprecision(1);

    this->compliance = 0;

    if ( 0 < this->markers )
    {
        this->compliance = best_compliance;

        ss << endl << "<br/><br/>Harmonic number: " << this->markers;
        ss << endl << "<br/>Compliance value: " << setprecision(5) << this->compliance;

        switch(this->automarking)
        {
        case ManualMarkers: break; // ss << ", fetched " << fetched_heightmap_values << " values"; break;
        case AutoMarkerWorking:
            {
                ss << ", computing...";
#ifdef _DEBUG
                ss << " (" << fetched_heightmap_values << ")";
#endif
                break;
            }
        case AutoMarkerFinished: break; // ss << ", fetched " << fetched_heightmap_values << " values"; break;
        }

        ss << endl << "<br/>Fundamental frequency: " << setprecision(2) << (f/this->markers) << " Hz";
        ss << endl << "<br/>Note: " << toneName();
    }

    bool first = 0 == this->comment;
    comments_->setComment( this->pos(), ss.str(), &this->comment );
    if (first)
    {
        this->comment->thumbnail( true );
        this->comment_model = this->comment->modelp;
    }

    this->comment->model()->pos = Heightmap::Position(
            p.time - 0.01/render_view_->model->xscale*render_view_->model->_pz,
            p.scale);

    //QToolTip::showText( screen_pos.toPoint(), QString(), this ); // Force tooltip to change position even if the text is the same as in the previous tooltip
    //QToolTip::showText( screen_pos.toPoint(), QString::fromLocal8Bit(ss.str().c_str()), this );

    if ( first )
    {
        this->comment->resize( 440, 225 );
    }

    if (found_better)
    {
        ofstream selected_tone("selected_tone.m");
        selected_tone
                << "fundamental_frequency = " << setprecision(30) << this->pos_hz/this->markers << ";" << endl
                << "selected_tone_number = " << this->markers << ";" << endl
                << "frequencies = fundamental_frequency * (1:3*selected_tone_number);" << endl;
    }
}


void TooltipModel::
        toneName(std::string& primaryName, std::string& secondaryName, float& accuracy)
{
    //2^(n/12)*440
    float tva12 = powf(2.f, 1.f/12);
    float tonef = log(pos_hz/this->markers/440.f)/log(tva12) + 45; // 440 is tone number 45, given C1 as tone 0
    int tone0 = floor(tonef + 0.5);
    int tone1 = (tone0 == (int)floor(tonef)) ? tone0 + 1 : tone0 - 1;
    accuracy = fabs(tonef-tone0);
    int octave0 = 0;
    int octave1 = 0;
    while(tone0<0) { tone0 += 12; octave0--;}
    while(tone1<0) { tone1 += 12; octave1--;}
    octave0 += tone0/12 + 1;
    octave1 += tone1/12 + 1;
    const char name[][3] = {
        "C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"
    };
    stringstream a;
    primaryName = ((stringstream&)(a << name[tone0%12] << octave0)).str();
    a.clear();
    secondaryName = ((stringstream&)(a << name[tone1%12] << octave1)).str();
}


std::string TooltipModel::
        toneName()
{
    std::string primary, secondary;
    float accuracy;
    toneName( primary, secondary, accuracy );

    std::stringstream ss;
    ss << primary << " (" << setprecision(0) << fixed << accuracy*100 << "% " << secondary << ")";
    return ss.str();
}


std::string TooltipModel::
        automarkingStr()
{
    if (markers_auto == markers && automarking==ManualMarkers)
        automarking = AutoMarkerWorking;

    switch(automarking)
    {
    case ManualMarkers: return "Manual";
    case AutoMarkerWorking: return "Working";
    case AutoMarkerFinished: return "Automatic";
    default: return "Invalid value";
    }
}


class TooltipModel::FetchDataTransform: public TooltipModel::FetchData
{
public:
    FetchDataTransform( RenderModel* m, Tfr::Stft* stft, float t )
    {
        Signal::pOperation o = m->renderSignalTarget->source();
        Signal::IntervalType i = std::max(0.f, t) * o->sample_rate();
        unsigned w = stft->chunk_size();
        i = i / w * w;
        Signal::Interval I( i, i+w );
        Tfr::pChunk chunk = (*stft)( o->readFixedLength(I) );

        abslog.reset( new GpuCpuData<float>(
                0,
                chunk->transform_data->getNumberOfElements(),
                GpuCpuVoidData::CudaGlobal));

        float2* src = chunk->transform_data->getCpuMemory();
        float* dst = abslog->getCpuMemory();
        for (unsigned i=0; i<abslog->getNumberOfElements1D(); ++i)
        {
            float2& v = src[i];
            //dst[i] = sqrt(v.x*v.x + v.y*v.y);
            dst[i] = 0.4f*powf(v.x*v.x + v.y*v.y, 0.1);
            dst[i] *= dst[i];
        }

        fa = chunk->freqAxis;
    }

    FetchDataTransform( RenderModel* m, Tfr::Cepstrum* cepstrum, float t )
    {
        Signal::pOperation o = m->renderSignalTarget->source();
        Signal::IntervalType i = std::max(0.f, t) * o->sample_rate();
        unsigned w = cepstrum->chunk_size();
        i = i / w * w;
        Signal::Interval I( i, i+w );
        Tfr::pChunk chunk = (*cepstrum)( o->readFixedLength(I) );

        abslog.reset( new GpuCpuData<float>(
                0,
                chunk->transform_data->getNumberOfElements(),
                GpuCpuVoidData::CudaGlobal));

        float2* src = chunk->transform_data->getCpuMemory();
        float* dst = abslog->getCpuMemory();
        for (unsigned i=0; i<abslog->getNumberOfElements1D(); ++i)
        {
            float2& v = src[i];
            //dst[i] = sqrt(v.x*v.x + v.y*v.y);
            dst[i] = 0.4f*powf(v.x*v.x + v.y*v.y, 0.1);
            dst[i] *= dst[i];
        }

        fa = chunk->freqAxis;
    }

    FetchDataTransform( RenderModel* m, Tfr::Cwt* cwt, float t )
    {
        Signal::pOperation o = m->renderSignalTarget->source();
        float fs = o->sample_rate();
        Signal::IntervalType firstSample = std::max(0.f, t) * fs;
        Signal::IntervalType numberOfSamples = cwt->next_good_size( 1, fs);
        firstSample = firstSample/numberOfSamples*numberOfSamples;
        unsigned support = cwt->wavelet_time_support_samples( fs );
        Signal::Interval I = Signal::Intervals( firstSample, firstSample+numberOfSamples ).enlarge( support ).coveredInterval();
        Tfr::pChunk chunk = (*cwt)( o->readFixedLength(I) );

        Tfr::CwtChunk* cwtchunk = dynamic_cast<Tfr::CwtChunk*>( chunk.get() );
        unsigned N = 0;
        for (unsigned i=0; i < cwtchunk->chunks.size(); ++i)
        {
            N += cwtchunk->chunks[i]->nScales() - (i!=0);
        }

        abslog.reset( new GpuCpuData<float>(
                0,
                make_cudaExtent( N, 1, 1),
                GpuCpuVoidData::CudaGlobal));

        float* dst = abslog->getCpuMemory();
        unsigned k = 0;
        for (unsigned j=0; j < cwtchunk->chunks.size(); ++j)
        {
            float2* src = cwtchunk->chunks[j]->transform_data->getCpuMemory();

            unsigned stride = cwtchunk->chunks[j]->nSamples();
            src += (unsigned)((t*o->sample_rate() - cwtchunk->chunk_offset)/o->sample_rate());

            for (unsigned i=(j!=0); i<cwtchunk->chunks[j]->nScales(); ++i)
            {
                float2& v = src[i*stride];
                //dst[i] = sqrt(v.x*v.x + v.y*v.y);
                dst[k] = 0.4f*powf(v.x*v.x + v.y*v.y, 0.1);
                dst[k] *= dst[k];
                k++;
            }
        }

        fa = chunk->freqAxis;
    }

    virtual float operator()( float /*t*/, float hz, bool* is_valid_value )
    {
        float i = std::max( 0.f, fa.getFrequencyScalar( hz ) );
        float local_max;
        float v = quad_interpol(i, abslog->getCpuMemory(),
                                abslog->getNumberOfElements1D(), 1, &local_max);
        if (is_valid_value)
            *is_valid_value = true;
        return v;
    }

private:
    Tfr::FreqAxis fa;
    boost::scoped_ptr<GpuCpuData<float> > abslog;
};


class TooltipModel::FetchDataHeightmap: public TooltipModel::FetchData
{
public:
    FetchDataHeightmap( RenderView* render_view )
        :
        ref_(render_view_->model->collections[0].get()),
        render_view_(render_view)
    {
        ref_.block_index[0] = (unsigned)-1;
    }


    virtual float operator()( float t, float hz, bool* is_valid_value )
    {
        Heightmap::Position p(t, 0);
        p.scale = render_view_->model->display_scale().getFrequencyScalar( hz );
        // Use quadratic interpolation to fetch estimates at given scale
        float value = render_view_->getHeightmapValue( p, &ref_, 0, true, is_valid_value );
        value*=value;
        return value;
    }

private:
    Heightmap::Reference ref_;
    RenderView* render_view_;
};


boost::shared_ptr<TooltipModel::FetchData> TooltipModel::FetchData::
        createFetchData( RenderView* view, float t )
{
    boost::shared_ptr<FetchData> r;
    Tfr::pTransform transform = view->model->collections[0]->transform();
    if (Tfr::Stft* stft = dynamic_cast<Tfr::Stft*>(transform.get()))
        r.reset( new FetchDataTransform( view->model, stft, t ) );
    else if (Tfr::Cwt* cwt = dynamic_cast<Tfr::Cwt*>(transform.get()))
        r.reset( new FetchDataTransform( view->model, cwt, t ) );
    else if (Tfr::Cepstrum* cepstrum = dynamic_cast<Tfr::Cepstrum*>(transform.get()))
        r.reset( new FetchDataTransform( view->model, cepstrum, t ) );
    else
    {
        BOOST_ASSERT( false );
        r.reset( new FetchDataHeightmap( view ) );
    }

    return r;
}


unsigned TooltipModel::
        guessHarmonicNumber( const Heightmap::Position& pos, float& best_compliance )
{
    TaskTimer tt("TooltipController::guessHarmonicNumber (%g, %g)", pos.time, pos.scale);
    double max_s = 0;
    unsigned max_i = 0;
    const Tfr::FreqAxis& display_scale = render_view_->model->display_scale();

    double F_min = 8;
    double F = display_scale.getFrequency( pos.scale );

    boost::shared_ptr<FetchData> fetcher = FetchData::createFetchData( render_view_, pos.time );

    fetched_heightmap_values = 0;
    unsigned n_tests = F/F_min;
    for (unsigned i=1; i<n_tests; ++i)
    {
        double s = computeMarkerMeasure(pos, i, fetcher.get());
        if (max_s < s)
        {
            max_s = s;
            max_i = i;
        }

        if ( 0<= last_fetched_scale_ )
        {
            break;
        }
    }

    //if (max_i<=1)
        //return 0;

    tt.info("%g Hz is harmonic number number %u, fundamental frequency is %g Hz. Did %u tests",
        F, max_i, F/max_i, n_tests);
    best_compliance = max_s;
    return max_i;
}


float TooltipModel::
      computeMarkerMeasure(const Heightmap::Position& pos, unsigned i, FetchData* fetcher)
{
    boost::shared_ptr<FetchData> myfetcher;
    if (0==fetcher)
    {
        myfetcher = FetchData::createFetchData( render_view_, pos.time );
        fetcher = myfetcher.get();
    }

    const Tfr::FreqAxis& display_scale = render_view_->model->display_scale();
    double F = display_scale.getFrequency( pos.scale );
    double F_top = display_scale.getFrequency(1.f);
    double penalty = 0.95;
    //double penalty = 0.97;
    // penalty should be in the range (0,1]
    // Close to 1 means that it's more likely to find few harmonies
    // Close to 0 means that it's more likely to find many harmonies

    double k = pow((double)i, -penalty);
    Heightmap::Position p = pos;

    double s = 0;
    double fundamental_f = F/i;
    unsigned count = std::min(F_top, 3*F)/fundamental_f;
    bool all_valid = true;
    for (unsigned j=1; j<=count; ++j)
    {
        double f = fundamental_f * j;
        bool is_valid_value;
        float value = (*fetcher)( pos.time, f, &is_valid_value );

        s += value * k;

        if (is_valid_value)
            ++fetched_heightmap_values;

        last_fetched_scale_ = p.scale;

        all_valid &= is_valid_value;

        if (!is_valid_value)
            last_fetched_scale_ = p.scale;
    }

    if (all_valid)
        last_fetched_scale_ = -1;
    else
    {
        if (AutoMarkerFinished == automarking)
            automarking = AutoMarkerWorking;
    }

    return s;
}


Heightmap::Position TooltipModel::
        pos()
{
    Heightmap::Position p( pos_time, render_view_->model->display_scale().getFrequencyScalar( pos_hz ));
    return p;
}

} // namespace Tools
