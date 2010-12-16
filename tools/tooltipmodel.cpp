#include "tooltipmodel.h"

namespace Tools {

TooltipModel::TooltipModel()
    :
        frequency(-1),
        max_so_far(-1),
        markers(0),
        comment(0),
        automarkers(false)
{
}


const Heightmap::Position& TooltipModel::
        comment_pos()
{
    return comment->model->pos;
}

} // namespace Tools
