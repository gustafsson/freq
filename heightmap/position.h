#ifndef HEIGHTMAPPOSITION_H
#define HEIGHTMAPPOSITION_H

namespace Heightmap {

class Position {
public:
    // TODO rename to normalized frequency or something...
    float time, scale;

    Position():time(0), scale(0) { }
    Position(float time, float scale):time(time), scale(scale) {}
};

} // Heightmap

#endif // HEIGHTMAPPOSITION_H
