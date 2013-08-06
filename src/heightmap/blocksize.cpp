#include "blocksize.h"

#include "collection.h"
#include "tfr/transform.h"
#include "signal/operation.h"

namespace Heightmap {

BlockSize::
        BlockSize(int texels_per_row, int texels_per_column)
    :
        texels_per_column_( texels_per_column ),
        texels_per_row_( texels_per_row )
{
    EXCEPTION_ASSERT_LESS( 1, texels_per_row );
    EXCEPTION_ASSERT_LESS( 1, texels_per_column );
}


bool BlockSize::
        operator==(const BlockSize& b)
{
    return texels_per_column_ == b.texels_per_column_ &&
            texels_per_row_ == b.texels_per_row_;
}


bool BlockSize::
        operator!=(const BlockSize& b)
{
    return texels_per_column_ != b.texels_per_column_ ||
           texels_per_row_ != b.texels_per_row_;
}


std::string BlockSize::
        toString() const
{
    std::stringstream ss;
    ss << "BlockSize(" << texels_per_row_ << ", " << texels_per_column_ << ")";
    return ss.str();
}

} // namespace Heightmap
