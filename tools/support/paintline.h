#ifndef PAINTLINE_H
#define PAINTLINE_H

#include "heightmap/position.h"

namespace Tools {
namespace Support {

class PaintLine
{
public:
    static void drawSlice(unsigned N, Heightmap::Position*pts, float r=0, float g=0, float b=0, float a=0.5);
};


} // namespace Support
} // namespace Tools

#endif // PAINTLINE_H
