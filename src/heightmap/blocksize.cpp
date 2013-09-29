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


BlockLayout::
        BlockLayout(BlockSize bs, SampleRate fs)
    :
      block_size_(bs),
      targetSampleRate_(fs)
{
    EXCEPTION_ASSERT_LESS( 0, fs );
}


BlockSize BlockLayout::
        block_size() const
{
    return block_size_;
}


SampleRate BlockLayout::
        targetSampleRate() const
{
    return targetSampleRate_;
}


bool BlockLayout::
        operator==(const BlockLayout& b)
{
    return block_size_ == b.block_size_ &&
            targetSampleRate_ == b.targetSampleRate_;
}


bool BlockLayout::
        operator!=(const BlockLayout& b)
{
    return !(*this == b);
}

} // namespace Heightmap


namespace Heightmap {

void BlockSize::
        test()
{
    // It should describe the size in texels of blocks in a heightmap
    {
        BlockSize b(12, 34);
        EXCEPTION_ASSERT_EQUALS(b.texels_per_block (), 12*34);
        EXCEPTION_ASSERT_EQUALS(b.texels_per_row (), 12);
        EXCEPTION_ASSERT_EQUALS(b.texels_per_column (), 34);
        EXCEPTION_ASSERT_EQUALS((boost::format("%1%")%b).str(), "BlockSize(12, 34)");
        EXCEPTION_ASSERT_EQUALS(b, BlockSize(12, 34));
    }

    // It should be immutable POD
    {
        // Implemented by no setters, only getters
    }
}


void BlockLayout::
        test()
{
    // It should describe the sizes of blocks
    {
        BlockLayout b(BlockSize(2,3),4);
        EXCEPTION_ASSERT_EQUALS(b.block_size (), BlockSize(2,3));
        EXCEPTION_ASSERT_EQUALS(b.targetSampleRate (), 4);
    }

    // It should be immutable POD
    {
        // Implemented by no setters, only getters
    }
}

} // namespace Heightmap
