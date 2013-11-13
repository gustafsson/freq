#ifndef FFTCLFFT_H
#define FFTCLFFT_H

#include "fftimplementation.h"

namespace Tfr {
    class FftClFft: public FftImplementation {
    public:
        void compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction );
        void computeR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output );
        void computeC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output );

        void compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, DataStorageSize n, FftDirection direction );
        void compute( DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize n );
        void inverse( Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n );
    };
}

#endif // STFT_CLFFT_H
