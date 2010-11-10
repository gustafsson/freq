#ifndef HEIGHTMAPPOSITION_H
#define HEIGHTMAPPOSITION_H

#include <tvector.h>

namespace Heightmap {

class Position {
public:
    // TODO rename to normalized frequency or something...
    float time, scale;

    Position():time(0), scale(0) { }
    Position(float time, float scale):time(time), scale(scale) {}

    tvector<2, float> operator()() { return tvector<2, float>(time, scale); }
};

} // Heightmap

#endif // HEIGHTMAPPOSITION_H
