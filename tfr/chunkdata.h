#ifndef CHUNKDATA_H
#define CHUNKDATA_H

#include "datastorage.h"
#include <complex>

namespace Tfr
{

typedef std::complex<float> ChunkElement;
typedef DataStorage<ChunkElement> ChunkData;

} // namespace Tfr

#endif // CHUNKDATA_H
