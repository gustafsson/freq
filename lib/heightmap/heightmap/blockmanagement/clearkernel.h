#ifndef CLEARKERNEL_H
#define CLEARKERNEL_H

#include "datastorage.h"

typedef DataStorage<float> BlockData;

extern "C"
void blockClearPart( BlockData::ptr block,
                 int start_t );

#endif // CLEARKERNEL_H
