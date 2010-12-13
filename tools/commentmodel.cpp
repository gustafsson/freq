#include "commentmodel.h"
#include <vector_functions.h>

namespace Tools
{

CommentModel::CommentModel()
    :
        scroll_scale(1),
        window_size(make_uint2(200,100)),
        freezed_position(false)
{
}


} // namespace Tools
