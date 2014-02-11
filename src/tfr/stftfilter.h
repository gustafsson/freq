#ifndef TFRSTFTFILTER_H
#define TFRSTFTFILTER_H

#include "tfr/chunkfilter.h"

namespace Tfr {


class StftFilterDesc : public Tfr::ChunkFilterDesc
{
public:
    StftFilterDesc();

    void transformDesc( Tfr::pTransformDesc m );
};


} // namespace Tfr

#endif // TFRSTFTFILTER_H
