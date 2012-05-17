#ifndef NORMALIZEKERNEL_H
#define NORMALIZEKERNEL_H

#include "datastorage.h"

void normalizedata( DataStorage<float>::Ptr data, int radius );
void normalizeTruncatedMean( DataStorage<float>::Ptr data, int radius, float truncation = 1.f );

#endif // NORMALIZEKERNEL_H
