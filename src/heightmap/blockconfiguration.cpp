#include "blockconfiguration.h"

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


} // namespace Heightmap
