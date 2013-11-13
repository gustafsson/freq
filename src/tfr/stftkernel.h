#ifndef STFTKERNEL_H
#define STFTKERNEL_H

#include "tfr/chunkdata.h"

template<typename T>
void        stftNormalizeInverse( boost::shared_ptr<DataStorage<T> > wave, unsigned length );
void        stftNormalizeInverse( Tfr::ChunkData::Ptr inwave, DataStorage<float>::Ptr outwave, unsigned length );
inline void stftDiscardImag( Tfr::ChunkData::Ptr inwavep, DataStorage<float>::Ptr outwavep )
{
    stftNormalizeInverse(inwavep, outwavep, 1);
}

void        stftToComplex( DataStorage<float>::Ptr inwavep, Tfr::ChunkData::Ptr outwavep );
void        cepstrumPrepareCepstra( Tfr::ChunkData::Ptr chunk, float normalization );
void        stftAverage( Tfr::ChunkData::Ptr input, Tfr::ChunkData::Ptr output, unsigned scales );

#endif // STFTKERNEL_H
