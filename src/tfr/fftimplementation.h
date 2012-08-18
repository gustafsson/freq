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
        virtual ~FftImplementation() {}

        /**
         * @brief compute Computes a fast fourier transform of input and stores
         * the results in output.
         *
         * @param input Input data. Data in a DataStorage is arranged row-wise.
         * input->size().width must be a value returned by sChunkSizeG() or
         * lChunkSizeS(). If input->size().height or input->size().depth is
         * larger than 1 that extra data is discarded.
         *
         * @param output Place to store output data, output->size().width must
         * be identical to input->size().width.
         *
         * @param direction Whether to do a forward och inverse(backward)
         * fourier transform.
         */
        virtual void compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, FftDirection direction ) = 0;
        /// @see compute( Tfr::ChunkData::Ptr, Tfr::ChunkData::Ptr, FftDirection )
        virtual void computeR2C( DataStorage<float>::Ptr input, Tfr::ChunkData::Ptr output ) = 0;
        /// @see compute( Tfr::ChunkData::Ptr, Tfr::ChunkData::Ptr, FftDirection )
        virtual void computeC2R( Tfr::ChunkData::Ptr input, DataStorage<float>::Ptr output ) = 0;

        /**
         * @brief compute Computes multiple fast fourier transforms of input
         * and stores the results in output. 'n' describes the number of
         * transforms and the size of each.
         *
         * @param input Input Data.
         *
         * @param output Place to store output data, output->numberOfElements() must be identical to input->numberOfElements()
         *
         * @param n n.width is the number of elements in each transform, this
         * must be a value returned by sChunkSizeG() or lChunkSizeS(). n.height
         * is the number of transforms. n.depth is not used.
         *
         * @param direction Whether to do a forward och inverse(backward) fourier transform.
         */
        virtual void compute( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, DataStorageSize n, FftDirection direction ) = 0;
        /// @see compute( Tfr::ChunkData::Ptr, Tfr::ChunkData::Ptr, DataStorageSize, FftDirection )
        virtual void compute( DataStorage<float>::Ptr inputbuffer, Tfr::ChunkData::Ptr transform_data, DataStorageSize n ) = 0;
        /// @see compute( Tfr::ChunkData::Ptr, Tfr::ChunkData::Ptr, DataStorageSize, FftDirection )
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
