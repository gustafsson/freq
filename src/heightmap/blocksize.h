#ifndef HEIGHTMAP_BLOCKCONFIGURATION_H
#define HEIGHTMAP_BLOCKCONFIGURATION_H

#include "tfr/freqaxis.h"
#include "amplitudeaxis.h"
#include "signal/intervals.h"

#include <boost/shared_ptr.hpp>

namespace Heightmap {

class BlockSize {
public:
    BlockSize(int texels_per_row, int texels_per_column);

    int texels_per_row() const { return texels_per_row_; }
    int texels_per_column() const { return texels_per_column_; }
    int texels_per_block() const { return texels_per_row() * texels_per_column(); }

    bool operator==(const BlockSize& b);
    bool operator!=(const BlockSize& b);

    std::string toString() const;

    template< class ostream_t > inline
    friend ostream_t& operator<<(ostream_t& os, const BlockSize& r) {
        return os << r.toString();
    }
private:
    int texels_per_column_;
    int texels_per_row_;
};


} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKCONFIGURATION_H
