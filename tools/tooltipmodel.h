#ifndef TOOLTIPMODEL_H
#define TOOLTIPMODEL_H

#include <QPointer>

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
    QPointer<CommentView> comment;
    bool automarkers;
};

} // namespace Tools

#endif // TOOLTIPMODEL_H
