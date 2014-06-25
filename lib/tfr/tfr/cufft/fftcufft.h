#ifndef FFTCUFFT_H
#define FFTCUFFT_H

#include "fftimplementation.h"

namespace Tfr {
    class FftCufft: public FftImplementation {
    public:
        void compute( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, FftDirection direction );
        void computeR2C( DataStorage<float>::ptr input, Tfr::ChunkData::ptr output );
        void computeC2R( Tfr::ChunkData::ptr input, DataStorage<float>::ptr output );

        void compute( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, DataStorageSize n, FftDirection direction );
        void compute( DataStorage<float>::ptr inputbuffer, Tfr::ChunkData::ptr transform_data, DataStorageSize actualSize );
        void inverse( Tfr::ChunkData::ptr inputdata, DataStorage<float>::ptr outputdata, DataStorageSize n );

        unsigned sChunkSizeG(unsigned x, unsigned multiple=1);
        unsigned lChunkSizeS(unsigned x, unsigned multiple=1);
    };
}

#endif // STFT_CUFFT_H
