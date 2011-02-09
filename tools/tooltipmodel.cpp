#include "tooltipmodel.h"

#include "sawe/project.h"
#include "tfr/cwt.h"
#include "tfr/cwtfilter.h"

#include "commentcontroller.h"
#include "heightmap/renderer.h"

// gpumisc
#include <TaskTimer.h>

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
        frequency(-1),
        max_so_far(-1),
        markers(0),
        comment(0),
        automarking(TooltipModel::ManualMarkers),
        comments_(0),
        render_view_(0),
        fetched_heightmap_values(0)
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
        showToolTip(Heightmap::Position p )
{
    Tfr::Cwt& c = Tfr::Cwt::Singleton();

    if (TooltipModel::ManualMarkers != this->automarking)
    {
        // computeMarkerMeasure and others will set automarking back to working if it is not finished
        this->automarking = AutoMarkerFinished;
    }

    fetched_heightmap_values = 0;

    // Fetch local max based on quadratic interpolation (if any max exists
    // within '1 pixel') to create a more accurate position. Move 'p' to this
    // local max if any is found.
    bool is_valid_value;
    float val = render_view_->getHeightmapValue( p, 0, &p.scale, false, &is_valid_value );
    if (!is_valid_value && TooltipModel::ManualMarkers != this->automarking)
        automarking = AutoMarkerWorking;

    bool found_better = val > max_so_far;
    if (found_better)
        this->pos = p;
    else
        p = this->pos;

    float FS = render_view_->model->project()->worker.source()->sample_rate();
    float f = c.compute_frequency2( FS, p.scale );
    float std_t = c.morlet_sigma_samples( FS, f ) / FS;
    float std_f = c.morlet_sigma_f( f );

    if (found_better || TooltipModel::ManualMarkers != this->automarking )
    {
        this->markers = this->markers_auto = guessHarmonicNumber( p );
    }

    if (found_better)
        this->max_so_far = val;

    stringstream ss;
    ss << setiosflags(ios::fixed)
       << "Time: " << setprecision(3) << p.time << " s<br/>"
       << "Frequency: " << setprecision(1) << f << " Hz<br/>";

    if (dynamic_cast<Tfr::CwtFilter*>(
            dynamic_cast<Signal::PostSink*>(render_view_->model->postsink().get())
            ->sinks()[0]->source().get()))
       ss << "Morlet standard deviation: " << setprecision(3) << std_t << " s, " << setprecision(1) << std_f << " Hz<br/>";

    ss << "Value here: " << setprecision(10) << this->max_so_far << setprecision(1);

    this->compliance = 0;
    this->frequency = f;

    if ( 0 < this->markers )
    {
        this->compliance = computeMarkerMeasure(p, this->markers);

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
    comments_->setComment( this->pos, ss.str(), &this->comment );
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
                << "fundamental_frequency = " << setprecision(30) << this->frequency/this->markers << ";" << endl
                << "selected_tone_number = " << this->markers << ";" << endl
                << "frequencies = fundamental_frequency * (1:3*selected_tone_number);" << endl;
    }
}


void TooltipModel::
        toneName(std::string& primaryName, std::string& secondaryName, float& accuracy)
{
    //2^(n/12)*440
    float tva12 = powf(2.f, 1.f/12);
    float tonef = log(frequency/this->markers/440.f)/log(tva12) + 45; // 440 is tone number 45, given C1 as tone 0
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


unsigned TooltipModel::
        guessHarmonicNumber( const Heightmap::Position& pos )
{
    TaskTimer tt("TooltipController::guessHarmonicNumber (%g, %g)", pos.time, pos.scale);
    double max_s = 0;
    unsigned max_i = 0;
    const Tfr::FreqAxis& display_scale = render_view_->model->display_scale();

    double F_min = 8;
    double F = display_scale.getFrequency( pos.scale );
    Heightmap::Reference ref(render_view_->model->collections[0].get());
    ref.block_index[0] = (unsigned)-1;

    fetched_heightmap_values = 0;
    unsigned n_tests = F/F_min;
    for (unsigned i=1; i<n_tests; ++i)
    {
        double s = computeMarkerMeasure(pos, i, &ref);
        if (max_s < s)
        {
            max_s = s;
            max_i = i;
        }

        if (AutoMarkerWorking == automarking)
        {
            break;
        }
    }

    //if (max_i<=1)
        //return 0;

    tt.info("%g Hz is harmonic number number %u, fundamental frequency is %g Hz. Did %u tests",
        F, max_i, F/max_i, n_tests);
    return max_i;
}


float TooltipModel::
        computeMarkerMeasure(const Heightmap::Position& pos, unsigned i, Heightmap::Reference* ref)
{
    Heightmap::Reference myref( render_view_->model->collections[0].get());
    if (0==ref)
    {
        ref = &myref;
        myref.block_index[0] = (unsigned)-1;
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
    for (unsigned j=1; j<=count; ++j)
    {
        double f = fundamental_f * j;
        p.scale = display_scale.getFrequencyScalar( f );
        // Use quadratic interpolation to fetch estimates at given scale
        bool is_valid_value;
        float value = render_view_->getHeightmapValue( p, ref, 0, true, &is_valid_value );
        value*=value;

        s += value * k;

        if (is_valid_value)
            ++fetched_heightmap_values;

        if (!is_valid_value && AutoMarkerFinished == automarking)
        {
            automarking = AutoMarkerWorking;
            return s;
        }
    }

    return s;
}

} // namespace Tools
