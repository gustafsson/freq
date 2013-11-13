#ifndef TVECTORSTRING_H
#define TVECTORSTRING_H

#include "tvector.h"

#include <iostream>


template<int N, typename T, typename Tb>
std::ostream& operator<<(std::ostream& o, const tvector<N, T, Tb>& v) {
    o << "(";
    for (int i=0; i<N; i++) {
        if (i>0)
            o << ", ";
        o << v[i];
    }
    o << ")";
    return o;
}


#endif // TVECTORSTRING_H
