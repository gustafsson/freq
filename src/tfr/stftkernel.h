#ifndef STFTKERNEL_H
#define STFTKERNEL_H

#include "tfr/chunkdata.h"

template<typename T>
void        stftNormalizeInverse( boost::shared_ptr<DataStorage<T> > wave, unsigned length );
void        stftNormalizeInverse( Tfr::ChunkData::ptr inwave, DataStorage<float>::ptr outwave, unsigned length );
inline void stftDiscardImag( Tfr::ChunkData::ptr inwavep, DataStorage<float>::ptr outwavep )
{
    stftNormalizeInverse(inwavep, outwavep, 1);
}

void        stftToComplex( DataStorage<float>::ptr inwavep, Tfr::ChunkData::ptr outwavep );
void        cepstrumPrepareCepstra( Tfr::ChunkData::ptr chunk, float normalization );
void        stftAverage( Tfr::ChunkData::ptr input, Tfr::ChunkData::ptr output, unsigned scales );

#endif // STFTKERNEL_H
