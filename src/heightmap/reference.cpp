#include "reference.h"

#include "referenceinfo.h"

#include <sstream>

using namespace std;

namespace Heightmap {


bool Reference::
        operator==(const Reference &b) const
{
    return log2_samples_size == b.log2_samples_size
            && block_index == b.block_index;
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


string Reference::
        toString() const
{
    std::stringstream ss;
    ss << "(" << log2_samples_size[0] << ":" << log2_samples_size[1] << " "
            << block_index[0] << ":" << block_index[1]
            << ")";
    return ss.str();
}


unsigned Reference::
        frequency_resolution() const
{
    return 1 << -log2_samples_size[1];
}


} // namespace Heightmap
