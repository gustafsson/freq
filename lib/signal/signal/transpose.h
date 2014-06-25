#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include "datastorage.h"

namespace Signal {

    void transpose(DataStorage<float>* dest, DataStorage<float>* src);

} // namespace Signal

#endif // TRANSPOSE_H
