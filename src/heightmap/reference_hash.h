#ifndef REFERENCE_HASH_H
#define REFERENCE_HASH_H

#include "reference.h"

#include <boost/functional/hash.hpp>

namespace Heightmap
{

// found with adl
inline std::size_t hash_value(Reference const& ref)
{
    std::size_t seed = 0;
    boost::hash_combine(seed, ref.log2_samples_size[0]);
    boost::hash_combine(seed, ref.log2_samples_size[1]);
    boost::hash_combine(seed, ref.block_index[0]);
    boost::hash_combine(seed, ref.block_index[1]);
    return seed;
}

} // namespace Heightmap

#endif // REFERENCE_HASH_H
