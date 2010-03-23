#ifndef POSITION_H
#define POSITION_H
#include <tvector.h>

class Position {
public:
    float time, scale;

    Position():time(0), scale(0) { }
    Position(float time, float scale):time(time), scale(scale) {}

    tvector<2, float> operator()() { return tvector<2, float>(time, scale); }
};

#endif