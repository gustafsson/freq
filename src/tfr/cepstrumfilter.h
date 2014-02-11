#ifndef CEPSTRUMFILTER_H
#define CEPSTRUMFILTER_H

#include "tfr/chunkfilter.h"

namespace Tfr {


class CepstrumFilterDesc : public Tfr::ChunkFilterDesc
{
public:
    CepstrumFilterDesc();

    void transformDesc( Tfr::pTransformDesc m );
};


} // namespace Tfr


#endif // CEPSTRUMFILTER_H
