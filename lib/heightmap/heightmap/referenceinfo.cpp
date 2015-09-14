#include "referenceinfo.h"
#include "exceptionassert.h"
#include "log.h"

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
        getOverlapping(const Reference& ref) const
{
    const Region texelCenter = getVisible (ref);
    Position a = texelCenter.a, b = texelCenter.b;

    // Compute the size of a texel
    float dt = (b.time-a.time) / (block_size_.visible_texels_per_row ());
    float ds = (b.scale-a.scale) / (block_size_.visible_texels_per_column ());

    // Remove the margin
    float m = block_size_.margin ();
    a.time -= dt*m;
    a.scale -= ds*m;
    b.time += dt*m;
    b.scale += ds*m;

//    // At {dt,ds}*0.5 to point at the center of texels
//    a.time += dt*0.5;
//    a.scale += ds*0.5;
//    b.time -= dt*0.5;
//    b.scale -= ds*0.5;

    return Region(a,b);
}


Region RegionFactory::
        getVisible(const Reference& ref) const
{
    Position a, b;
    // For integers 'i': "2 to the power of 'i'" == powf(2.f, i) == ldexpf(1.f, i)
    Position blockSize( ldexp (1. , ref.log2_samples_size[0]),
                        ldexpf(1.f, ref.log2_samples_size[1]));
    a.time = blockSize.time * ref.block_index[0];
    a.scale = blockSize.scale * ref.block_index[1];
    b.time = a.time + blockSize.time;
    b.scale = a.scale + blockSize.scale;

    return Region(a,b);
}


ReferenceInfo::
        ReferenceInfo(const Reference& reference, const BlockLayout& block_layout, const VisualizationParams::const_ptr& visualization_params)
    :
      block_layout_(block_layout),
      visualization_params_(visualization_params),
      reference_(reference),
      visible_region_(RegionFactory(block_layout).getVisible (reference_))
{
}


long double ReferenceInfo::
        sample_rate() const
{
    return block_layout_.visible_texels_per_row () / visible_region_.time ();
}


bool ReferenceInfo::
        containsPoint(Position p) const
{
    return visible_region_.a.time <= p.time && p.time <= visible_region_.b.time &&
            visible_region_.a.scale <= p.scale && p.scale <= visible_region_.b.scale;
}


// returns false if the given BoundsCheck is out of bounds
bool ReferenceInfo::
        boundsCheck(BoundsCheck c) const
{
    FreqAxis cfa = visualization_params_->display_scale();
    float ahz = cfa.getFrequency(visible_region_.a.scale);
    float bhz = cfa.getFrequency(visible_region_.b.scale);

    if (c & ReferenceInfo::BoundsCheck_HighS)
    {
        // Check the frequency resolution, if it is low enough we're out-of-bounds

        // Assuming that the frequency resolution is either not-growing or not-shrinking,
        // it is enough to check the end-points as they will be extrema.
        float scaledelta = visible_region_.scale()/block_layout_.visible_texels_per_column ();
        float a2hz = cfa.getFrequency(visible_region_.a.scale + scaledelta);
        float b2hz = cfa.getFrequency(visible_region_.b.scale - scaledelta);

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
        float tdelta = visible_region_.time()/block_layout_.visible_texels_per_row ();

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
    ss      << "T[" << visible_region_.a.time << ":" << visible_region_.b.time << ") "
            << "F[" << visible_region_.a.scale << ":" << visible_region_.b.scale << ") "
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
    Region overlapping = RegionFactory(block_layout_).getOverlapping (reference_);

    long double FS = block_layout_.targetSampleRate();

    Signal::Interval i( floor(overlapping.a.time * FS), ceil(overlapping.b.time * FS) );
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
