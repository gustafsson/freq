#include "referenceinfo.h"

#include <sstream>

using namespace std;

namespace Heightmap {

ReferenceInfo::
        ReferenceInfo(const BlockConfiguration& block_config, const Reference& reference)
    :
      block_config_(block_config),
      reference_(reference)
{
}


Region ReferenceInfo::
        getRegion() const
{
    return reference_.getRegion(block_config_.samplesPerBlock(), block_config_.scalesPerBlock());
}


long double ReferenceInfo::
        sample_rate() const
{
    Region r = getRegion();
    return ldexp(1.0, -reference_.log2_samples_size[0]) - 1/(long double)r.time();
}


bool ReferenceInfo::
        containsPoint(Position p) const
{
    Region r = getRegion();

    return r.a.time <= p.time && p.time <= r.b.time &&
            r.a.scale <= p.scale && p.scale <= r.b.scale;
}


// returns false if the given BoundsCheck is out of bounds
bool ReferenceInfo::
        boundsCheck(BoundsCheck c, const Tfr::TransformDesc* transform, float length) const
{
    Region r = getRegion();

    const Tfr::FreqAxis& cfa = block_config_.display_scale();
    float ahz = cfa.getFrequency(r.a.scale);
    float bhz = cfa.getFrequency(r.b.scale);

    if (c & Reference::BoundsCheck_HighS)
    {
        float scaledelta = (r.scale())/block_config_.scalesPerBlock();
        float a2hz = cfa.getFrequency(r.a.scale + scaledelta);
        float b2hz = cfa.getFrequency(r.b.scale - scaledelta);

        const Tfr::FreqAxis& tfa = transformScale (transform);
        float scalara = tfa.getFrequencyScalar(ahz);
        float scalarb = tfa.getFrequencyScalar(bhz);
        float scalara2 = tfa.getFrequencyScalar(a2hz);
        float scalarb2 = tfa.getFrequencyScalar(b2hz);

        if (fabsf(scalara2 - scalara) < 0.5f && fabsf(scalarb2 - scalarb) < 0.5f )
            return false;
    }

    if (c & Reference::BoundsCheck_HighT)
    {
        float atres = displayedTimeResolution (ahz, transform);
        float btres = displayedTimeResolution (bhz, transform);
        float tdelta = 2*r.time()/block_config_.samplesPerBlock();
        if (btres > tdelta && atres > tdelta)
            return false;
    }

    if (c & Reference::BoundsCheck_OutT)
    {
        if (r.a.time >= length )
            return false;
    }

    if (c & Reference::BoundsCheck_OutS)
    {
        if (r.a.scale >= 1)
            return false;

        if (r.b.scale > 1)
            return false;
    }

    return true;
}


bool ReferenceInfo::
        tooLarge(float length) const
{
    Region r = getRegion();
    if (r.b.time > 2 * length && r.b.scale > 2 )
        return true;
    return false;
}


std::string ReferenceInfo::
        toString() const
{
    Region r = getRegion();
    stringstream ss;
    ss << "(" << r.a.time << ":" << r.b.time << " " << r.a.scale << ":" << r.b.scale << " "
            << getInterval() << " "
            << reference_.log2_samples_size[0] << ":" << reference_.log2_samples_size[1] << " "
            << reference_.block_index[0] << ":" << reference_.block_index[1]
            << ")";
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
    int samplesPerBlock = block_config_.samplesPerBlock ();
    long double blockSize = samplesPerBlock * ldexp(1.f,reference_.log2_samples_size[0]);
    long double elementSize = 1.0 / sample_rate();
    long double blockLocalSize = samplesPerBlock * elementSize;

    // where the first element starts
    long double startTime = blockSize * reference_.block_index[0] - elementSize*.5f;

    // where the last element ends
    long double endTime = startTime + blockLocalSize;

    long double FS = block_config_.targetSampleRate ();
    Signal::Interval i( max(0.L, floor(startTime * FS)), ceil(endTime * FS) );

    //Position a, b;
    //getArea( a, b );
    //Signal::SamplesIntervalDescriptor::Interval i = { a.time * FS, b.time * FS };
    return i;
}


Signal::Interval ReferenceInfo::
        spannedElementsInterval(const Signal::Interval& I, Signal::Interval& spannedBlockSamples) const
{
    unsigned samples_per_block = block_config_.samplesPerBlock();
    long double blockSize = samples_per_block * ldexp(1.,reference_.log2_samples_size[0]);
    long double FS = block_config_.targetSampleRate ();

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

    long double
        a = floor(localStartTime*p),
        b = ceil(localEndTime*p);

    Signal::Interval r(
            max(0.L, a),
            max(0.L, b));

    return r&I;
}


Reference ReferenceInfo::
        reference() const
{
    return this->reference_;
}


Tfr::FreqAxis ReferenceInfo::
        transformScale(const Tfr::TransformDesc* transform) const
{
    return transform->freqAxis(block_config_.targetSampleRate ());
}


float ReferenceInfo::
        displayedTimeResolution(float hz, const Tfr::TransformDesc* transform) const
{
    return transform->displayedTimeResolution(block_config_.targetSampleRate (), hz);
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
