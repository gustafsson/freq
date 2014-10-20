#include "referenceinfo.h"
#include "exceptionassert.h"

#include <sstream>

using namespace std;

namespace Heightmap {

RegionFactory::
        RegionFactory(const BlockLayout& block_size)
    :
      block_size_(block_size)
{

}


Region RegionFactory::
        operator()(const Reference& ref, bool render_region) const
{
    Position a, b;
    // For integers 'i': "2 to the power of 'i'" == powf(2.f, i) == ldexpf(1.f, i)
    Position blockSize( ldexp (1. , ref.log2_samples_size[0]),
                        ldexpf(1.f, ref.log2_samples_size[1]));
    a.time = blockSize.time * ref.block_index[0];
    a.scale = blockSize.scale * ref.block_index[1];
    b.time = a.time + blockSize.time;
    b.scale = a.scale + blockSize.scale;

    if (!render_region)
    {
        float dt = ldexpf(.5f, block_size_.mipmaps ())/block_size_.visible_texels_per_row ();
        float ds = ldexpf(.5f, block_size_.mipmaps ())/block_size_.visible_texels_per_column ();
        dt *= b.time-a.time;
        ds *= b.scale-a.scale;

        if (0==block_size_.mipmaps ())
            dt = ds = 0.f;

        Region r(a,b);
        a.time -= dt;
        a.scale -= ds;
        b.time += dt;
        b.scale += ds;
    }

    return Region(a,b);
}


ReferenceInfo::
        ReferenceInfo(const Reference& reference, const BlockLayout& block_layout, const VisualizationParams::const_ptr& visualization_params)
    :
      block_layout_(block_layout),
      visualization_params_(visualization_params),
      reference_(reference),
      r(RegionFactory(block_layout)(reference_))
{
}


Region ReferenceInfo::
        region() const
{
    return r;
}


long double ReferenceInfo::
        sample_rate() const
{
    return ldexp(1.0, -reference_.log2_samples_size[0])*block_layout_.visible_texels_per_row () - 1/(long double)r.time();
}


bool ReferenceInfo::
        containsPoint(Position p) const
{
    return r.a.time <= p.time && p.time <= r.b.time &&
            r.a.scale <= p.scale && p.scale <= r.b.scale;
}


// returns false if the given BoundsCheck is out of bounds
bool ReferenceInfo::
        boundsCheck(BoundsCheck c) const
{
    FreqAxis cfa = visualization_params_->display_scale();
    float ahz = cfa.getFrequency(r.a.scale);
    float bhz = cfa.getFrequency(r.b.scale);

    if (c & ReferenceInfo::BoundsCheck_HighS)
    {
        // Check the frequency resolution, if it is low enough we're out-of-bounds

        // Assuming that the frequency resolution is either not-growing or not-shrinking,
        // it is enough to check the end-points as they will be extrema.
        float scaledelta = r.scale()/block_layout_.visible_texels_per_column ();
        float a2hz = cfa.getFrequency(r.a.scale + scaledelta);
        float b2hz = cfa.getFrequency(r.b.scale - scaledelta);

        float scalara = displayedFrequencyResolution(ahz, a2hz);
        float scalarb = displayedFrequencyResolution(bhz, b2hz);

        // if the number of data points between two adjacent texels in this region is less than one
        // then there are no more details to be seen by zooming in, we are thus out-of-bounds
        if (fabsf(scalara) <= 1.0f && fabsf(scalarb) <= 1.0f )
            return false;
    }

    if (c & ReferenceInfo::BoundsCheck_HighT)
    {
        // ["time units" / "1 data point"]
        float atres = displayedTimeResolution (ahz);
        float btres = displayedTimeResolution (bhz);

        // ["time units" / "1 texel"].
        float tdelta = r.time()/block_layout_.visible_texels_per_row ();

        // [data points / texel]
        atres = tdelta/atres;
        btres = tdelta/btres;

        if (fabsf(atres) <= 1.0f && fabsf(btres) <= 1.0f )
            return false;
    }

//    if (c & ReferenceInfo::BoundsCheck_OutT)
//    {
//        if (r.a.time >= tfr_mapping_.length )
//            return false;
//    }

//    if (c & ReferenceInfo::BoundsCheck_OutS)
//    {
//        if (r.a.scale >= 1)
//            return false;

//        if (r.b.scale > 1)
//            return false;
//    }

    return true;
}


std::string ReferenceInfo::
        toString() const
{
    stringstream ss;
    ss      << "T[" << r.a.time << ":" << r.b.time << ") "
            << "F[" << r.a.scale << ":" << r.b.scale << ") "
            << "S" << getInterval();
    return ss.str();
}


Signal::Interval ReferenceInfo::
        getInterval() const
{
    // TaskTimer tt("Reference::getInterval");

    // Similiar to getArea, but uses 1./sample_rate() instead of
    // "2 ^ log2_samples_size[0]" to compute the actual size of this block.
    // blockSize refers to the non-overlapping size.

    // Overlapping is needed to compute the same result for block borders
    // between two adjacent blocks. Thus the interval of samples that affect
    // this block overlap slightly into the samples that are needed for the
    // next block.
    Region data(RegionFactory(block_layout_)(reference_,false));
    long double elementSize = 1.0 / sample_rate();
    data.a.time -= elementSize*.5;
    data.b.time += elementSize*.5;

    long double FS = block_layout_.targetSampleRate();

    Signal::Interval i( floor(data.a.time * FS), ceil(data.b.time * FS) );
    return i;
}


Signal::Interval ReferenceInfo::
        spannedElementsInterval(const Signal::Interval& I, Signal::Interval& spannedBlockSamples) const
{
    EXCEPTION_ASSERTX (false, "not implemented");

    unsigned samples_per_block = block_layout_.texels_per_row ();
    long double blockSize = samples_per_block * ldexp(1.,reference_.log2_samples_size[0]);
    long double FS = block_layout_.targetSampleRate();

    unsigned stepsPerBlock = samples_per_block - 1;
    long double p = FS*blockSize/stepsPerBlock;
    double localStartTime = I.first/p;
    double localEndTime = I.last/p;
    double s = 0.; // .5/stepsPerBlock;

    // round off to spanned elements
    if (0 != localStartTime)
        localStartTime = ceil(localStartTime + .5 - s) - .5;
    localEndTime = floor(localEndTime - .5 + s) + .5;

    // didn't even span one element, expand to span the elements it intersects with
    if ( localEndTime - localStartTime < (0 != localStartTime ? 1.0 : 0.5 ))
    {
        // this is only an accetable fallback if blocks can't be bigger than I.count()
        // BlockFilter::largestApplied takes care of this

        double middle = floor( (I.first/p + I.last/p)*0.5 );
        localStartTime = max(0.0, middle - 0.5);
        localEndTime = middle + 1.5;
    }

    Signal::IntervalType first = stepsPerBlock * reference_.block_index[0];
    spannedBlockSamples.first = min((double)samples_per_block, max(0.0, ceil(localStartTime) - first));  // Interval::first is inclusive
    spannedBlockSamples.last = min((double)samples_per_block, max(0.0, ceil(localEndTime) - first)); // Interval::last is exclusive

    // TODO make this work
    spannedBlockSamples.first = 0;
    spannedBlockSamples.last = samples_per_block;

    long double
        a = floor(localStartTime*p),
        b = ceil(localEndTime*p);

    Signal::Interval r(
            max(0.L, min((long double)Signal::Interval::IntervalType_MAX, a)),
            max(0.L, min((long double)Signal::Interval::IntervalType_MAX, b)));

    return r&I;
}


Reference ReferenceInfo::
        reference() const
{
    return this->reference_;
}


float ReferenceInfo::
        displayedTimeResolution(float hz) const
{
    return visualization_params_->detail_info()->displayedTimeResolution(block_layout_.targetSampleRate(), hz);
}


float ReferenceInfo::
        displayedFrequencyResolution(float hz1, float hz2 ) const
{
    return visualization_params_->detail_info()->displayedFrequencyResolution(block_layout_.targetSampleRate(), hz1, hz2);
}


void ReferenceInfo::
        test()
{
//#ifdef _DEBUG
//if ((r&I) != r && localEndTime-localStartTime > 2.f)
//{
//    int stopp = 1;
//}
//#endif
}

} // namespace Heightmap
