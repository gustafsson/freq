#include "heightmap/reference.h"
#include "heightmap/collection.h"

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
    // For integers 'i': "2 to the power of 'i'" == powf(2.f, i) == ldexpf(1.f, i)
    Position blockSize( _collection->samples_per_block() * ldexpf(1.f,log2_samples_size[0]),
                        _collection->scales_per_block() * ldexpf(1.f,log2_samples_size[1]));
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
        containsSpectrogram() const
{
    Position a, b;
    getArea( a, b );

    if (b.time-a.time < _collection->min_sample_size().time*_collection->samples_per_block() )
        return false;
    //float msss = _collection->min_sample_size().scale;
    //unsigned spb = _collection->_scales_per_block;
    //float ms = msss*spb;
    if (b.scale-a.scale < _collection->min_sample_size().scale*_collection->scales_per_block() )
        return false;

    float length = _collection->target->length();
    if (a.time >= length )
        return false;

    if (a.scale >= 1)
        return false;

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

std::string Reference::
        toString() const
{
    Position a, b;
    getArea( a, b );
    std::stringstream ss;
    ss << "(" << a.time << " " << a.scale << ";" << b.time << " " << b.scale << " ! "
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
    float blockSize = _collection->samples_per_block() * ldexpf(1.f,log2_samples_size[0]);
    float blockLocalSize = (_collection->samples_per_block()-1) / sample_rate();

    float startTime = blockSize * block_index[0];
    float endTime = startTime + blockLocalSize;

    float FS = _collection->target->sample_rate();
    Signal::Interval i( startTime * FS, endTime * FS+1 );

    //Position a, b;
    //getArea( a, b );
    //Signal::SamplesIntervalDescriptor::Interval i = { a.time * FS, b.time * FS };
    return i;
}

float Reference::
        sample_rate() const
{
    Position a, b;
    getArea( a, b );
    return ldexpf(1.f, -log2_samples_size[0]) - 1/(b.time-a.time);
}

unsigned Reference::
        frequency_resolution() const
{
    return 1 << -log2_samples_size[1];
}
} // namespace Heightmap
