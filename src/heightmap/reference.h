#ifndef HEIGHTMAPREFERENCE_H
#define HEIGHTMAPREFERENCE_H

#include "tvector.h"

#include <string>

namespace Heightmap {

class Reference {
public:
    typedef tvector<2,int> Scale;
    typedef tvector<2,unsigned> Index;

    Scale log2_samples_size;
    Index block_index;

    Reference();

    bool operator==(const Reference &b) const;

    unsigned frequency_resolution() const;

    /** child references */
    Reference left() const;
    Reference right() const;
    Reference top() const;
    Reference bottom() const;

    /** sibblings, 3 other references who share the same parent */
    Reference sibbling1() const;
    Reference sibbling2() const;
    Reference sibbling3() const;

    /** sibblings */
    Reference sibblingLeft() const;
    Reference sibblingRight() const;
    Reference sibblingTop() const;
    Reference sibblingBottom() const;

    /** parent */
    Reference parent() const;
    Reference parentVertical() const;
    Reference parentHorizontal() const;

    std::string toString() const;

    template< class ostream_t > inline
    friend ostream_t& operator<<(ostream_t& os, const Reference& r) {
        return os << r.toString();
    }
};


} // namespace Heightmap

#endif // HEIGHTMAPREFERENCE_H
