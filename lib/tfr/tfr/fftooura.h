#ifndef FFTOOURA_H
#define FFTOOURA_H

#include "fftimplementation.h"
#include <vector>

namespace Tfr {
    class FftOoura: public FftImplementation {
    public:
        void compute( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, FftDirection direction ) override;
        void computeR2C( DataStorage<float>::ptr input, Tfr::ChunkData::ptr output ) override;
        void computeC2R( Tfr::ChunkData::ptr input, DataStorage<float>::ptr output ) override;

        void compute( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, DataStorageSize n, FftDirection direction ) override;
        void compute( DataStorage<float>::ptr inputbuffer, Tfr::ChunkData::ptr transform_data, DataStorageSize n ) override;
        void inverse( Tfr::ChunkData::ptr inputdata, DataStorage<float>::ptr outputdata, DataStorageSize n ) override;

        void computeOoura( Tfr::ChunkData::ptr input_output, FftDirection direction );
        void computeOoura( Tfr::ChunkData::ptr input_output, DataStorageSize n, FftDirection direction );
    private:
        std::vector<float> w;
        std::vector<int> ip;
    };
}

#endif // STFT_OOURA_H
