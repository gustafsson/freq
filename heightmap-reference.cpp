#include "heightmap-reference.h"
#include "heightmap-collection.h"

namespace Heightmap {

bool Reference::
        operator==(const Reference &b) const
{
    return log2_samples_size == b.log2_samples_size
            && block_index == b.block_index
            && _collection == b._collection;
}

void Reference::
        getArea( Position &a, Position &b) const
{
    Position blockSize( _collection->samples_per_block() * pow(2,log2_samples_size[0]),
                        _collection->scales_per_block() * pow(2,log2_samples_size[1]));
    a.time = blockSize.time * block_index[0];
    a.scale = blockSize.scale * block_index[1];
    b.time = a.time + blockSize.time;
    b.scale = a.scale + blockSize.scale;
}

/* child references */
Reference Reference::
        left()
{
    Reference r = *this;
    r.log2_samples_size[0]--;
    r.block_index[0]<<=1;
    return r;
}
Reference Reference::
        right()
{
    Reference r = *this;
    r.log2_samples_size[0]--;
    (r.block_index[0]<<=1)++;
    return r;
}
Reference Reference::
        top()
{
    Reference r = *this;
    r.log2_samples_size[1]--;
    r.block_index[1]<<=1;
    return r;
}
Reference Reference::
        bottom()
{
    Reference r = *this;
    r.log2_samples_size[1]--;
    (r.block_index[1]<<=1)++;
    return r;
}

/* sibblings, 3 other references who share the same parent */
Reference Reference::
        sibbling1()
{
    Reference r = *this;
    r.block_index[0]^=1;
    return r;
}
Reference Reference::
        sibbling2()
{
    Reference r = *this;
    r.block_index[1]^=1;
    return r;
}
Reference Reference::
        sibbling3()
{
    Reference r = *this;
    r.block_index[0]^=1;
    r.block_index[1]^=1;
    return r;
}

/* parent */
Reference Reference::parent() {
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

    Signal::pSource wf = _collection->worker->source();
    if (a.time >= wf->length() )
        return false;

    if (a.scale >= 1)
        return false;

    return true;
}

bool Reference::
        toLarge() const
{
    Position a, b;
    getArea( a, b );
    Signal::pSource wf = _collection->worker->source();
    if (b.time > 2 * wf->length() && b.scale > 2 )
        return true;
    return false;
}

unsigned Reference::
        samplesPerBlock() const
{
    return _collection->samples_per_block();
}

Signal::SamplesIntervalDescriptor::Interval Reference::
        getInterval()
{
    Position a,b;
    getArea(a,b);
    unsigned FS = _collection->worker->source()->sample_rate();
    Signal::SamplesIntervalDescriptor::Interval i = { a.time * FS, b.time*FS };
    return i;
}

} // namespace Heightmap
