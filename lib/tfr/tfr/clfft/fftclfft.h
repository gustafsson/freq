#ifndef FFTCLFFT_H
#define FFTCLFFT_H

#include "fftimplementation.h"

namespace Tfr {
    class FftClFft: public FftImplementation {
    public:
        void compute( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, FftDirection direction );
        void computeR2C( DataStorage<float>::ptr input, Tfr::ChunkData::ptr output );
        void computeC2R( Tfr::ChunkData::ptr input, DataStorage<float>::ptr output );

        void compute( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, DataStorageSize n, FftDirection direction );
        void compute( DataStorage<float>::ptr inputbuffer, Tfr::ChunkData::ptr transform_data, DataStorageSize n );
        void inverse( Tfr::ChunkData::ptr inputdata, DataStorage<float>::ptr outputdata, DataStorageSize n );
    };
}

#endif // STFT_CLFFT_H
