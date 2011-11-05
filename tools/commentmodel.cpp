#include "commentmodel.h"

namespace Tools
{

CommentModel::CommentModel()
    :
        scroll_scale(1),
        thumbnail(false),
        window_size(200,100),
        freezed_position(false),
        screen_pos(UpdateScreenPositionFromWorld,0)
{
}


} // namespace Tools
