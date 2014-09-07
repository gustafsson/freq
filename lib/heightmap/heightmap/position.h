#ifndef HEIGHTMAPPOSITION_H
#define HEIGHTMAPPOSITION_H

namespace Heightmap {

class Position {
public:
    // TODO rename to normalized frequency or something...
    double time; float scale;

    Position() : time(0), scale(0) {}
    Position(double time, float scale) : time(time), scale(scale) {}

    bool operator== (Position const&b) { return time==b.time && scale==b.scale; }
    bool operator!= (Position const&b) { return !(*this==b); }

    template< class ostream_t > inline
    friend ostream_t& operator<<(ostream_t& os, const Position& p) {
        return os << p.time << ":" << p.scale;
    }
};


class Region {
public:
    Position a, b;

    Region(Position a, Position b) : a(a), b(b) {}

    bool operator== (Region const&r) { return a==r.a && b==r.b; }
    bool operator!= (Region const&r) { return !(*this==r); }

    float time() const { return b.time - a.time; }
    float scale() const { return b.scale - a.scale; }

    template< class ostream_t > inline
    friend ostream_t& operator<<(ostream_t& os, const Region& r) {
        return os << "(t=" << r.a.time << ":" << r.b.time << " s=" << r.a.scale << ":" << r.b.scale << ")";
    }
};

} // Heightmap

#endif // HEIGHTMAPPOSITION_H
