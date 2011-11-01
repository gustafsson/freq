#include "commentmodel.h"
#include <vector_functions.h>

namespace Tools
{

CommentModel::CommentModel()
    :
        scroll_scale(1),
        thumbnail(false),
        window_size(make_uint2(200,100)),
        freezed_position(false),
        screen_pos(make_float2(UpdateScreenPositionFromWorld,0))
{
}


} // namespace Tools
