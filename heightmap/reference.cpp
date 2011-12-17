#include "heightmap/reference.h"
#include "heightmap/collection.h"

#include "signal/operation.h"
#include "tfr/transform.h"

using namespace std;

namespace Heightmap {

bool Reference::
        operator==(const Reference &b) const
{
    return log2_samples_size == b.log2_samples_size
            && block_index == b.block_index;
            // Don't compare _collection == b._collection;
}

void Reference::
        getArea( Position &a, Position &b) const
{
    getArea( a, b, samplesPerBlock(), scalesPerBlock() );
}

void Reference::
        getArea( Position &a, Position &b, unsigned samples_per_block, unsigned scales_per_block ) const
{
    // TODO make Referece independent of samples_per_block and scales_per_block
    // For integers 'i': "2 to the power of 'i'" == powf(2.f, i) == ldexpf(1.f, i)
    Position blockSize( samples_per_block * ldexpf(1.f,log2_samples_size[0]),
                        scales_per_block * ldexpf(1.f,log2_samples_size[1]));
    a.time = blockSize.time * block_index[0];
    a.scale = blockSize.scale * block_index[1];
    b.time = a.time + blockSize.time;
    b.scale = a.scale + blockSize.scale;
}

/* child references */
Reference Reference::
        left() const
{
    Reference r = *this;
    r.log2_samples_size[0]--;
    r.block_index[0]<<=1;
    return r;
}
Reference Reference::
        right() const
{
    Reference r = *this;
    r.log2_samples_size[0]--;
    (r.block_index[0]<<=1)++;
    return r;
}
Reference Reference::
        top() const
{
    Reference r = *this;
    r.log2_samples_size[1]--;
    (r.block_index[1]<<=1)++;
    return r;
}
Reference Reference::
        bottom() const
{
    Reference r = *this;
    r.log2_samples_size[1]--;
    r.block_index[1]<<=1;
    return r;
}

/* sibblings, 3 other references who share the same parent */
Reference Reference::
        sibbling1() const
{
    Reference r = *this;
    r.block_index[0]^=1;
    return r;
}
Reference Reference::
        sibbling2() const
{
    Reference r = *this;
    r.block_index[1]^=1;
    return r;
}
Reference Reference::
        sibbling3() const
{
    Reference r = *this;
    r.block_index[0]^=1;
    r.block_index[1]^=1;
    return r;
}

Reference Reference::
        sibblingLeft() const
{
    Reference r = *this;
    if(0<r.block_index[0])
        --r.block_index[0];
    return r;
}
Reference Reference::
        sibblingRight() const
{
    Reference r = *this;
    ++r.block_index[0];
    return r;
}
Reference Reference::
        sibblingTop() const
{
    Reference r = *this;
    ++r.block_index[1];
    return r;
}
Reference Reference::
        sibblingBottom() const
{
    Reference r = *this;
    if(0<r.block_index[1])
        --r.block_index[1];
    return r;
}

/* parent */
Reference Reference::parent() const {
    Reference r = *this;
    r.log2_samples_size[0]++;
    r.log2_samples_size[1]++;
    r.block_index[0]>>=1;
    r.block_index[1]>>=1;
    return r;
}

Reference::
        Reference(Collection *collection)
:   _collection(collection)
{}

bool Reference::
        containsPoint(Position p) const
{
    Position a, b;
    getArea( a, b );

    return a.time <= p.time && p.time <= b.time &&
            a.scale <= p.scale && p.scale <= b.scale;
}


bool Reference::
        boundsCheck(BoundsCheck c) const
{
    Position a, b;
    getArea( a, b );

    float FS = _collection->target->sample_rate();
    const Tfr::FreqAxis& cfa = _collection->display_scale();
    float ahz = cfa.getFrequency(a.scale);
    float bhz = cfa.getFrequency(b.scale);

    if (c & BoundsCheck_HighS)
    {
        float scaledelta = (b.scale-a.scale)/scalesPerBlock();
        float a2hz = cfa.getFrequency(a.scale + scaledelta);
        float b2hz = cfa.getFrequency(b.scale - scaledelta);

        const Tfr::FreqAxis& tfa = _collection->transform()->freqAxis(FS);
        float scalara = tfa.getFrequencyScalar(ahz);
        float scalarb = tfa.getFrequencyScalar(bhz);
        float scalara2 = tfa.getFrequencyScalar(a2hz);
        float scalarb2 = tfa.getFrequencyScalar(b2hz);

        if (fabsf(scalara2 - scalara) < 0.5f && fabsf(scalarb2 - scalarb) < 0.5f )
            return false;
    }

    if (c & BoundsCheck_HighT)
    {
        float atres = _collection->transform()->displayedTimeResolution(FS, ahz);
        float btres = _collection->transform()->displayedTimeResolution(FS, bhz);
        float tdelta = 2*(b.time-a.time)/samplesPerBlock();
        if (btres > tdelta && atres > tdelta)
            return false;
    }

    if (c & BoundsCheck_OutT)
    {
        float length = _collection->target->length();
        if (a.time >= length )
            return false;
    }

    if (c & BoundsCheck_OutS)
    {
        if (a.scale >= 1)
            return false;

        if (b.scale > 1)
            return false;
    }

    return true;
}


bool Reference::
        tooLarge() const
{
    Position a, b;
    getArea( a, b );
    Signal::pOperation wf = _collection->target;
    if (b.time > 2 * wf->length() && b.scale > 2 )
        return true;
    return false;
}

string Reference::
        toString() const
{
    Position a, b;
    getArea( a, b );
    stringstream ss;
    ss << "(" << a.time << " " << a.scale << ";" << b.time << " " << b.scale << " ! "
            << getInterval() << " ! "
            << log2_samples_size[0] << " " << log2_samples_size[1] << ";"
            << block_index[0] << " " << block_index[1]
            << ")";
    return ss.str();
}

unsigned Reference::
        samplesPerBlock() const
{
    return _collection->samples_per_block();
}

unsigned Reference::
        scalesPerBlock() const
{
    return _collection->scales_per_block();
}

Collection* Reference::
        collection() const
{
    return _collection;
}

void Reference::
        setCollection(Collection* c)
{
    _collection = c;
}

Signal::Interval Reference::
        getInterval() const
{
    // Similiar to getArea, but uses 1./sample_rate() instead of
    // "2 ^ log2_samples_size[0]" to compute the actual size of this block.
    // blockSize refers to the non-overlapping size.

    // Overlapping is needed to compute the same result for block borders
    // between two adjacent blocks. Thus the interval of samples that affect
    // this block overlap slightly into the samples that are needed for the
    // next block.
    long double blockSize = samplesPerBlock() * ldexp(1.f,log2_samples_size[0]);
    long double elementSize = 1.0 / sample_rate();
    long double blockLocalSize = samplesPerBlock() * elementSize;

    // where the first element starts
    long double startTime = blockSize * block_index[0] - elementSize*.5f;

    // where the last element ends
    long double endTime = startTime + blockLocalSize;

    long double FS = _collection->target->sample_rate();
    Signal::Interval i( max(0.L, floor(startTime * FS)), ceil(endTime * FS) );

    //Position a, b;
    //getArea( a, b );
    //Signal::SamplesIntervalDescriptor::Interval i = { a.time * FS, b.time * FS };
    return i;
}


Signal::Interval Reference::
        spannedElementsInterval(const Signal::Interval& I, Signal::Interval& spannedBlockSamples) const
{
    unsigned samples_per_block = samplesPerBlock();
    long double blockSize = samples_per_block * ldexp(1.,log2_samples_size[0]);
    long double FS = _collection->target->sample_rate();

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

    Signal::IntervalType first = stepsPerBlock * block_index[0];
    spannedBlockSamples.first = min((double)samples_per_block, max(0.0, ceil(localStartTime) - first));  // Interval::first is inclusive
    spannedBlockSamples.last = min((double)samples_per_block, max(0.0, ceil(localEndTime) - first)); // Interval::last is exclusive

    long double
        a = floor(localStartTime*p),
        b = ceil(localEndTime*p);

    Signal::Interval r(
            max(0.L, a),
            max(0.L, b));

#ifdef _DEBUG
    if ((r&I) != r && localEndTime-localStartTime > 2.f)
    {
        int stopp = 1;
    }
#endif

    return r&I;
}


long double Reference::
        sample_rate() const
{
    Position a, b;
    getArea( a, b );
    return ldexp(1.0, -log2_samples_size[0]) - 1/((long double)b.time-a.time);
}

unsigned Reference::
        frequency_resolution() const
{
    return 1 << -log2_samples_size[1];
}
} // namespace Heightmap
