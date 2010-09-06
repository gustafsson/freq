// TODO remove file


#include "inversecwt.h"
#include "wavelet.cu.h"


#include <CudaException.h>

//#define TIME_ICWT
#define TIME_ICWT if(0)

namespace Tfr {

InverseCwt::InverseCwt(cudaStream_t stream)
:   _stream(stream)
{
}

Signal::pBuffer InverseCwt::
operator()(Tfr::Chunk& chunk)
{

}

} // namespace Tfr
