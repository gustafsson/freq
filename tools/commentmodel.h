#ifndef COMMENTMODEL_H
#define COMMENTMODEL_H

#include <vector>
#include <string>
#include "heightmap/position.h"

namespace Tools
{

class CommentModel
{
public:
    CommentModel();

    struct Label
    {
        std::string text;
        Heightmap::Position pos;
    };

    std::vector<Label> labels;
};


} // namespace Tools

#endif // COMMENTMODEL_H
