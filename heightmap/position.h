#ifndef HEIGHTMAPPOSITION_H
#define HEIGHTMAPPOSITION_H

namespace Heightmap {

class Position {
public:
    // TODO rename to normalized frequency or something...
    float time, scale;

    Position() : time(0), scale(0) {}
    Position(float time, float scale) : time(time), scale(scale) {}

    bool operator== (Position const&b) { return time==b.time && scale==b.scale; }
    bool operator!= (Position const&b) { return !(*this==b); }
};

} // Heightmap

#endif // HEIGHTMAPPOSITION_H
