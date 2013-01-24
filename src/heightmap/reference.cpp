#include "heightmap/reference.h"
#include "heightmap/collection.h"

#include "signal/operation.h"
#include "tfr/transform.h"

using namespace std;

namespace Heightmap {


BlockConfiguration::
        BlockConfiguration( Collection* collection )
    :   collection_(collection)
{}


Collection* BlockConfiguration::
        collection() const
{
    return this->collection_;
}


void BlockConfiguration::
        setCollection(Collection* c)
{
    this->collection_ = c;
}


unsigned BlockConfiguration::
        samplesPerBlock() const
{
    return this->collection_->samples_per_block ();
}


unsigned BlockConfiguration::
        scalesPerBlock() const
{
    return this->collection_->scales_per_block ();
}


float BlockConfiguration::
        targetSampleRate() const
{
    return this->collection_->target->sample_rate ();
}


Tfr::FreqAxis BlockConfiguration::
        display_scale() const
{
    return this->collection_->display_scale ();
}


Tfr::FreqAxis BlockConfiguration::
        transform_scale() const
{
    return collection_->transform()->freqAxis(targetSampleRate ());
}


float BlockConfiguration::
        displayedTimeResolution(float hz) const
{
    return collection_->transform()->displayedTimeResolution(targetSampleRate (), hz);
}


float BlockConfiguration::
        length() const
{
    return collection_->target->length();
}


bool Reference::
        operator==(const Reference &b) const
{
    return log2_samples_size == b.log2_samples_size
            && block_index == b.block_index;
}

Region Reference::
        getRegion() const
{
    return getRegion(samplesPerBlock(), scalesPerBlock());
}

Region Reference::
        getRegion( unsigned samples_per_block, unsigned scales_per_block ) const
{
    Position a, b;
    // TODO make Referece independent of samples_per_block and scales_per_block
    // For integers 'i': "2 to the power of 'i'" == powf(2.f, i) == ldexpf(1.f, i)
    Position blockSize( samples_per_block * ldexpf(1.f,log2_samples_size[0]),
                        scales_per_block * ldexpf(1.f,log2_samples_size[1]));
    a.time = blockSize.time * block_index[0];
    a.scale = blockSize.scale * block_index[1];
    b.time = a.time + blockSize.time;
    b.scale = a.scale + blockSize.scale;

    return Region(a,b);
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

Reference Reference::parentVertical() const {
    Reference r = *this;
    r.log2_samples_size[1]++;
    r.block_index[1]>>=1;
    return r;
}

Reference Reference::parentHorizontal() const {
    Reference r = *this;
    r.log2_samples_size[0]++;
    r.block_index[0]>>=1;
    return r;
}

Reference::
        Reference(Collection *collection)
:   block_config_(new BlockConfiguration(collection))
{}


Reference::
        ~Reference()
{
}


bool Reference::
        containsPoint(Position p) const
{
    Region r = getRegion();

    return r.a.time <= p.time && p.time <= r.b.time &&
            r.a.scale <= p.scale && p.scale <= r.b.scale;
}


bool Reference::
        boundsCheck(BoundsCheck c) const
{
    Region r = getRegion();

    float FS = block_config_->targetSampleRate ();
    const Tfr::FreqAxis& cfa = block_config_->display_scale();
    float ahz = cfa.getFrequency(r.a.scale);
    float bhz = cfa.getFrequency(r.b.scale);

    if (c & BoundsCheck_HighS)
    {
        float scaledelta = (r.scale())/scalesPerBlock();
        float a2hz = cfa.getFrequency(r.a.scale + scaledelta);
        float b2hz = cfa.getFrequency(r.b.scale - scaledelta);

        const Tfr::FreqAxis& tfa = block_config_->transform_scale ();
        float scalara = tfa.getFrequencyScalar(ahz);
        float scalarb = tfa.getFrequencyScalar(bhz);
        float scalara2 = tfa.getFrequencyScalar(a2hz);
        float scalarb2 = tfa.getFrequencyScalar(b2hz);

        if (fabsf(scalara2 - scalara) < 0.5f && fabsf(scalarb2 - scalarb) < 0.5f )
            return false;
    }

    if (c & BoundsCheck_HighT)
    {
        float atres = block_config_->displayedTimeResolution (ahz);
        float btres = block_config_->displayedTimeResolution (bhz);
        float tdelta = 2*r.time()/samplesPerBlock();
        if (btres > tdelta && atres > tdelta)
            return false;
    }

    if (c & BoundsCheck_OutT)
    {
        float length = block_config_->length();
        if (r.a.time >= length )
            return false;
    }

    if (c & BoundsCheck_OutS)
    {
        if (r.a.scale >= 1)
            return false;

        if (r.b.scale > 1)
            return false;
    }

    return true;
}


bool Reference::
        tooLarge() const
{
    Region r = getRegion();
    if (r.b.time > 2 * block_config_->length () && r.b.scale > 2 )
        return true;
    return false;
}

string Reference::
        toString() const
{
    Region r = getRegion();
    stringstream ss;
    ss << "(" << r.a.time << ":" << r.b.time << " " << r.a.scale << ":" << r.b.scale << " "
            << getInterval() << " "
            << log2_samples_size[0] << ":" << log2_samples_size[1] << " "
            << block_index[0] << ":" << block_index[1]
            << ")";
    return ss.str();
}

unsigned Reference::
        samplesPerBlock() const
{
    return block_config_->samplesPerBlock ();
}

unsigned Reference::
        scalesPerBlock() const
{
    return block_config_->scalesPerBlock ();
}

Collection* Reference::
        collection() const
{
    return block_config_->collection();
}

void Reference::
        setCollection(Collection* c)
{
    block_config_->setCollection (c);
}

Signal::Interval Reference::
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
    long double blockSize = samplesPerBlock() * ldexp(1.f,log2_samples_size[0]);
    long double elementSize = 1.0 / sample_rate();
    long double blockLocalSize = samplesPerBlock() * elementSize;

    // where the first element starts
    long double startTime = blockSize * block_index[0] - elementSize*.5f;

    // where the last element ends
    long double endTime = startTime + blockLocalSize;

    long double FS = block_config_->targetSampleRate ();
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
    long double FS = block_config_->targetSampleRate ();

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
    Region r = getRegion();
    return ldexp(1.0, -log2_samples_size[0]) - 1/(long double)r.time();
}

unsigned Reference::
        frequency_resolution() const
{
    return 1 << -log2_samples_size[1];
}
} // namespace Heightmap
