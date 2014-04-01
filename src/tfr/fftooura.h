#ifndef FFTOOURA_H
#define FFTOOURA_H

#include "fftimplementation.h"
#include <vector>

namespace Tfr {
    class FftOoura: public FftImplementation {
    public:
        void compute( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, FftDirection direction );
        void computeR2C( DataStorage<float>::ptr input, Tfr::ChunkData::ptr output );
        void computeC2R( Tfr::ChunkData::ptr input, DataStorage<float>::ptr output );

        void compute( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, DataStorageSize n, FftDirection direction );
        void compute( DataStorage<float>::ptr inputbuffer, Tfr::ChunkData::ptr transform_data, DataStorageSize n );
        void inverse( Tfr::ChunkData::ptr inputdata, DataStorage<float>::ptr outputdata, DataStorageSize n );

    private:
        std::vector<float> w;
        std::vector<int> ip;
    };
}

#endif // STFT_OOURA_H
