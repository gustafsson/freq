#include "reference.h"

#include "referenceinfo.h"

using namespace std;

namespace Heightmap {


bool Reference::
        operator==(const Reference &b) const
{
    return log2_samples_size == b.log2_samples_size
            && block_index == b.block_index;
}

Region Reference::
        getRegion() const
{
    return ReferenceInfo(*block_config_.get (), *this).getRegion();
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

/*Reference::
        Reference(Collection *collection)
:   block_config_(new BlockConfiguration(collection))
{}*/


Reference::
        Reference( const BlockConfiguration& block_config )
    : block_config_(new BlockConfiguration(block_config))
{}


Reference::
        ~Reference()
{
}


bool Reference::
        containsPoint(Position p) const
{
    return ReferenceInfo(*block_config_.get (), *this).containsPoint(p);
}


bool Reference::
        boundsCheck(BoundsCheck c, const Tfr::TransformDesc* transform, float length) const
{
    return ReferenceInfo(*block_config_.get (), *this).boundsCheck(c, transform, length);
}


bool Reference::
        tooLarge(float length) const
{
    return ReferenceInfo(*block_config_.get (), *this).tooLarge(length);
}

string Reference::
        toString() const
{
    return ReferenceInfo(*block_config_.get (), *this).toString();
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

Signal::Interval Reference::
        getInterval() const
{
    return ReferenceInfo(*block_config_.get (), *this).getInterval();
}


Signal::Interval Reference::
        spannedElementsInterval(const Signal::Interval& I, Signal::Interval& spannedBlockSamples) const
{
    return ReferenceInfo(*block_config_.get (), *this).spannedElementsInterval (I, spannedBlockSamples);
}


long double Reference::
        sample_rate() const
{
    return ReferenceInfo(*block_config_.get (), *this).sample_rate();
}

unsigned Reference::
        frequency_resolution() const
{
    return 1 << -log2_samples_size[1];
}


} // namespace Heightmap
