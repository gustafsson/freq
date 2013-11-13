#ifndef TMATRIXSTRING_H
#define TMATRIXSTRING_H

#include "tmatrix.h"
#include "tvectorstring.h"


template<int N, typename T, int M>
std::ostream& operator<<(std::ostream& o, const tmatrix<N, T, M>& v) {
    o << "{";
    for (int i=0; i<M; i++) {
        if (i>0)
            o << "; ";
        o << v[i];
    }
    o << "}";
    return o;
}


#endif // TMATRIXSTRING_H
