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


class Region {
public:
    Position a, b;

    Region(Position a, Position b) : a(a), b(b) {}

    bool operator== (Region const&r) { return a==r.a && b==r.b; }
    bool operator!= (Region const&r) { return !(*this==r); }

    float time() const { return b.time - a.time; }
    float scale() const { return b.scale - a.scale; }
};

} // Heightmap

#endif // HEIGHTMAPPOSITION_H
