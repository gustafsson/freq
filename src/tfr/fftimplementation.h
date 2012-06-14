#ifndef FFTIMPLEMENTATION_H
#define FFTIMPLEMENTATION_H

#include "tfr/chunkdata.h"

namespace Tfr {
    enum FftDirection
    {
        FftDirection_Forward = -1,
        FftDirection_Inverse = 1
    };

    class FftImplementation {
    public:
        static FftImplementation& Singleton();

        virtual void compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction ) = 0;
        virtual void computeR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output ) = 0;
        virtual void computeC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output ) = 0;

        virtual void compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, DataStorageSize n, FftDirection direction ) = 0;
        virtual void compute( DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize n ) = 0;
        virtual void inverse( Tfr::ChunkData::Ptr inputdata, DataStorage<float>::Ptr outputdata, DataStorageSize n ) = 0;

        /**
          Returns the smallest ok chunk size strictly greater than x that also is
          a multiple of 'multiple'.
          'multiple' must be a power of 2.
          */
        virtual unsigned sChunkSizeG(unsigned x, unsigned multiple=1);

        /**
          Returns the largest ok chunk size strictly smaller than x that also is
          a multiple of 'multiple'.
          'multiple' must be a power of 2.
          */
        virtual unsigned lChunkSizeS(unsigned x, unsigned multiple=1);
    };
}

#endif // STFTIMPLEMENTATION_H
