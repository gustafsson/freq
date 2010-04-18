#ifndef TFRINVERSECWT_H
#define TFRINVERSECWT_H

#include "tfr-chunk.h"
#include "signal-source.h"
#include <boost/shared_ptr.hpp>
#include "tfr-filter.h"

namespace Tfr {

class InverseCwt
{
public:
    InverseCwt(cudaStream_t stream=0);

    Signal::pBuffer operator()(Tfr::Chunk&);

    pFilter filter;
private:
    cudaStream_t _stream;
};

} // namespace Tfr

#endif // TFRINVERSECWT_H
