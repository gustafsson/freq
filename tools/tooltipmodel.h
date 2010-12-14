#ifndef TOOLTIPMODEL_H
#define TOOLTIPMODEL_H

#include "commentview.h"

namespace Tools {

class TooltipModel
{
public:
    TooltipModel();

    const Heightmap::Position& comment_pos();

    Heightmap::Position pos;
    float frequency;
    float max_so_far;
    unsigned markers;
    CommentView* comment;
    bool automarkers;
};

} // namespace Tools

#endif // TOOLTIPMODEL_H
