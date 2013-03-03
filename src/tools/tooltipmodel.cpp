#include "tooltipmodel.h"

#include "sawe/project.h"
#include "tfr/cwt.h"
#include "tfr/stft.h"
#include "tfr/cwtfilter.h"
#include "tfr/stftfilter.h"
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
#include <iomanip>

using namespace std;

namespace Tools {

TooltipModel::TooltipModel()
    :
        pos_time(0),
        pos_hz(-1),
        max_so_far(-1),
        compliance(-1),
        markers(0),
        markers_auto(0),
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
        showToolTip( Heightmap::Position p, bool adjustScaleToLocalPeak )
{
    EXCEPTION_ASSERT( render_view_ );

    switch(this->automarking)
    {
    case TooltipModel::ManualMarkers:
        break;

    case TooltipModel::NoMarkers:
        max_so_far = -1; // new positions are always accepted
        break;

    case TooltipModel::AutoMarkerFinished:
        break;

    case TooltipModel::AutoMarkerWorking:
        // computeMarkerMeasure and others will set automarking back to working if it is not finished
        this->automarking = AutoMarkerFinished;
        break;
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
    float val = 0;
    if (last_fetched_scale_is_valid )
        val = render_view_->getHeightmapValue( p, 0, adjustScaleToLocalPeak?&p.scale:0,
                                               false, &is_valid_value );
	// elseor
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
    {
        p = this->pos();
        EXCEPTION_ASSERT( this->markers );
    }

    float FS = render_view_->model->project()->worker.source()->sample_rate();
    float f = this->pos_hz;

    float best_compliance = 0;

    switch(this->automarking)
    {
    case TooltipModel::ManualMarkers:
        if (found_better)
            this->markers = this->markers_auto = guessHarmonicNumber( p, best_compliance );
        else
            best_compliance = computeMarkerMeasure( p, this->markers, 0);
        break;

    case TooltipModel::NoMarkers:
        this->markers = 0;
        break;

    case TooltipModel::AutoMarkerFinished:
    case TooltipModel::AutoMarkerWorking:
        this->markers = this->markers_auto = guessHarmonicNumber( p, best_compliance );
        break;
    }

    if (found_better)
        this->max_so_far = val;

    stringstream ss;

    float std_t = 0;
    float std_f = 0;

    if (dynamic_cast<Tfr::CwtFilter*>( render_view_->model->block_filter()))
    {
        const Tfr::Cwt* c = render_view_->model->getParam<Tfr::Cwt>();
        std_t = c->morlet_sigma_samples( FS, f ) / FS;
        std_f = c->morlet_sigma_f( f );
    }
    else if (dynamic_cast<Tfr::StftFilter*>( render_view_->model->block_filter()))
    {
        const Tfr::StftDesc* f = render_view_->model->getParam<Tfr::StftDesc>();
        std_t = f->chunk_size() / FS / 2;
        std_f = FS / f->chunk_size() / 2;
    }

    ss << setiosflags(ios::fixed);
    int timeprecision = std::max(0, (int)floor(-log10(std_t)) + 2); // display 2 significant digits in std_t (same decimal precision in p.time)
    int freqprecision = std::max(0, (int)floor(-log10(std_f)) + 2); // display 2 significant digits in std_f (same decimal precision in f)
    if (std_t != 0)
        ss << "Time: " << setprecision(timeprecision) << p.time << " \u00B1 " << " " << setprecision(timeprecision) << std_t << " s<br/>";
    else
        ss << "Time: " << setprecision(timeprecision) << p.time << " s<br/>";

    if (std_f != 0)
        ss << "Frequency: " << setprecision(freqprecision) << f << " \u00B1 " << setprecision(freqprecision) << std_f << " Hz<br/>";
    else
        ss << "Frequency: " << setprecision(freqprecision) << f << " Hz<br/>";

    int valueprecision = std::max(0, (int)floor(-log10(max_so_far)) + 3); // display 3 significant digits in value
    ss << "Value here: " << setprecision(valueprecision) << this->max_so_far << setprecision(1);

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
        case NoMarkers: break;
        }

        ss << endl << "<br/>Fundamental frequency: " << setprecision(2) << (f/this->markers) << " Hz";
        ss << endl << "<br/>Note: " << toneName();
    }

    bool first = 0 == this->comment;

    comments_->setComment( this->pos(), ss.str(), &this->comment );
    EXCEPTION_ASSERT(this->comment);
    if (first)
    {
        this->comment->thumbnail( true );
        this->comment_model = this->comment->modelp;
    }

    if (TooltipModel::NoMarkers == this->automarking)
        this->comment->thumbnail( false );

    this->comment->model()->pos = p;

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
    stringstream s0, s1;
    s0 << name[tone0%12] << octave0;
    s1 << name[tone1%12] << octave1;

    primaryName = s0.str();
    secondaryName = s1.str();
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
    FetchDataTransform( RenderModel* m, const Tfr::StftDesc* stft, float t )
    {
        Signal::pOperation o = m->renderSignalTarget->source();
        Signal::IntervalType i = std::max(0.f, t) * o->sample_rate();
        unsigned w = stft->chunk_size();
        i = i / w * w;
        Signal::Interval I( i, i+w );

        // Only check the first channel
        // TODO check other channels
        Tfr::pChunk chunk = (*stft->createTransform())( o->readFixedLength(I)->getChannel (0));

        abslog.reset( new DataStorage<float>( chunk->transform_data->size() ));

        Tfr::ChunkElement* src = chunk->transform_data->getCpuMemory();
        float* dst = abslog->getCpuMemory();
        for (unsigned i=0; i<abslog->numberOfElements(); ++i)
        {
            Tfr::ChunkElement& v = src[i];
            dst[i] = abs(v);
            //dst[i] = 0.4f*powf(norm(v), 0.1);
            //dst[i] *= dst[i];
        }

        fa = chunk->freqAxis;
    }

    FetchDataTransform( RenderModel* m, const Tfr::CepstrumDesc* cepstrum, float t )
    {
        Signal::pOperation o = m->renderSignalTarget->source();
        Signal::IntervalType i = std::max(0.f, t) * o->sample_rate();
        unsigned w = cepstrum->chunk_size();
        i = i / w * w;
        Signal::Interval I( i, i+w );
        // Only check the first channel
        // TODO check other channels
        Tfr::pChunk chunk = (*cepstrum->createTransform())( o->readFixedLength(I)->getChannel (0));

        abslog.reset( new DataStorage<float>(chunk->transform_data->size()));

        Tfr::ChunkElement* src = chunk->transform_data->getCpuMemory();
        float* dst = abslog->getCpuMemory();
        for (unsigned i=0; i<abslog->numberOfElements(); ++i)
        {
            Tfr::ChunkElement& v = src[i];
            dst[i] = abs(v);
            //dst[i] = 0.4f*powf(v.x*v.x + v.y*v.y, 0.1);
            //dst[i] *= dst[i];
        }

        fa = chunk->freqAxis;
    }

    FetchDataTransform( RenderModel* m, const Tfr::Cwt* cwt, float t )
    {
        Signal::pOperation o = m->renderSignalTarget->source();
        float fs = o->sample_rate();

        Tfr::DummyCwtFilter filter;
        filter.source(o);
        filter.transform ( cwt->createTransform ());
        Signal::IntervalType sample = std::max(0.f, t) * fs;
        const Signal::Interval I = filter.requiredInterval (Signal::Interval(sample, sample+1), filter.transform ());
        // Only check the first channel
        // TODO check other channels
        Tfr::pChunk chunk = (*filter.transform ())(o->readFixedLength (I)->getChannel (0));

        Tfr::CwtChunk* cwtchunk = dynamic_cast<Tfr::CwtChunk*>( chunk.get() );
        unsigned N = 0;
        for (unsigned i=0; i < cwtchunk->chunks.size(); ++i)
            N += cwtchunk->chunks[i]->nScales() - (i!=0);

        EXCEPTION_ASSERT( N == cwt->nScales( fs ) );

        abslog.reset( new DataStorage<float>(N));

        float* dst = abslog->getCpuMemory();
        unsigned k = 0;
        for (unsigned j=0; j < cwtchunk->chunks.size(); ++j)
        {
            Tfr::CwtChunkPart* chunkpart = dynamic_cast<Tfr::CwtChunkPart*>(cwtchunk->chunks[j].get());
            Tfr::ChunkElement* src = chunkpart->transform_data->getCpuMemory();

            unsigned stride = chunkpart->nSamples();
            unsigned scales = chunkpart->nScales();
            double scale = chunkpart->original_sample_rate / chunkpart->sample_rate;

            // see chunkpart->getInterval()
            Signal::Interval chunkInterval(
                    chunkpart->chunk_offset.asFloat()*scale + 0.5,
                    (chunkpart->chunk_offset + stride).asFloat()*scale + 0.5);

            Signal::IntervalType sample = t*fs;
            if (chunkInterval.first > sample)
                sample = chunkInterval.first;
            if (chunkInterval.last <= sample)
                sample = chunkInterval.last-1;

            unsigned x = (sample - chunkInterval.first) / scale;
            if (x >= stride)
                x = stride - 1;

            for (unsigned i=(j!=0); i<scales; ++i)
            {
                Tfr::ChunkElement& v = src[i*stride + x];
                dst[k++] = abs(v);
            }
        }

        EXCEPTION_ASSERT( k == cwt->nScales(fs) );

        fa = chunk->freqAxis;
    }

    virtual float operator()( float /*t*/, float hz, bool* is_valid_value )
    {
        float i = std::max( 0.f, fa.getFrequencyScalar( hz ) );
        float local_max;
        float v = quad_interpol(i, abslog->getCpuMemory(),
                                abslog->numberOfElements(), 1, &local_max);
        if (is_valid_value)
            *is_valid_value = true;
        return v;
    }

    virtual float nextFrequency( float hz )
    {
        float i = std::max( 0.f, fa.getFrequencyScalar( hz ) );
        return fa.getFrequency(i+1);
    }

private:
    Tfr::FreqAxis fa;
    DataStorage<float>::Ptr abslog;
};


class TooltipModel::FetchDataHeightmap: public TooltipModel::FetchData
{
public:
    FetchDataHeightmap( RenderView* render_view )
        :
        ref_(render_view_->model->block_configuration()),
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

    virtual float nextFrequency( float hz )
    {
        return hz * 1.02f;
    }

private:
    Heightmap::Reference ref_;
    RenderView* render_view_;
};


boost::shared_ptr<TooltipModel::FetchData> TooltipModel::FetchData::
        createFetchData( RenderView* view, float t )
{
    boost::shared_ptr<FetchData> r;
    const Tfr::TransformDesc* transform = view->model->collections[0]->transform();
    if (const Tfr::CepstrumDesc* cepstrum = dynamic_cast<const Tfr::CepstrumDesc*>(transform))
        r.reset( new FetchDataTransform( view->model, cepstrum, t ) );
    else if (const Tfr::StftDesc* stft = dynamic_cast<const Tfr::StftDesc*>(transform))
        r.reset( new FetchDataTransform( view->model, stft, t ) );
    else if (const Tfr::Cwt* cwt = dynamic_cast<const Tfr::Cwt*>(transform))
        r.reset( new FetchDataTransform( view->model, cwt, t ) );
    else
    {
        return r;
        //r.reset( new FetchDataHeightmap( view ) );
    }

    return r;
}


unsigned TooltipModel::
        guessHarmonicNumber( const Heightmap::Position& pos, float& best_compliance )
{
    TaskTimer tt("TooltipController::guessHarmonicNumber (%g, %g)", pos.time, pos.scale);

    double max_s = -1;
    unsigned max_i = 1;
    fetched_heightmap_values = 0;

    const Tfr::FreqAxis& display_scale = render_view_->model->display_scale();
    boost::shared_ptr<FetchData> fetcher = FetchData::createFetchData( render_view_, pos.time );
    if (!fetcher)
        return 0;

    double F = display_scale.getFrequency( pos.scale );
    double F2 = fetcher->nextFrequency( F );
    F = std::min(F, F2);
    unsigned n_tests = F/(F2-F)/3;

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

    tt.info("%g Hz is harmonic number number %u, fundamental frequency is %g Hz. Did %u tests",
        F, max_i, F/max_i, n_tests);
    best_compliance = max_s;
    EXCEPTION_ASSERT( 0 < max_i );
    return max_i;
}


float TooltipModel::
      computeMarkerMeasure(const Heightmap::Position& pos, unsigned i, FetchData* fetcher)
{
    EXCEPTION_ASSERT( 0 < i );
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
