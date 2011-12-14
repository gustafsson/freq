#ifndef BLOCK_H
#define BLOCK_H

#include "reference.h"

// gpumisc
#include "datastorage.h"

// boost
#include <boost/shared_ptr.hpp>


namespace Heightmap {

    class GlBlock;

    // TODO it would probably look awesome if new blocks weren't displayed
    // instantaneously but rather faded in from 0 or from their previous value.
    // This method could be used to slide between the images of two different
    // signals or channels as well. This should be implemented by rendering two or
    // more separate collections in Heightmap::Renderer. It would fetch Blocks by
    // their 'Reference' from the different collections and use a shader to
    // transfer results between them.
    class Block {
    public:
        Block( Reference ref );
        ~Block();

        // TODO move this value to a complementary class
        unsigned frame_number_last_used;

        // Zoom level for this slot, determines size of elements
        Reference ref;
        boost::shared_ptr<GlBlock> glblock;

        typedef DataStorage<float>::Ptr pData;

        /**
            TODO test this in a multi gpu environment
            For multi-GPU or (just multithreaded) environments, each GPU-thread have
            its own cuda context and data can't be  transfered between cuda contexts
            without first going to the cpu. Therefore a 'cpu_copy' is kept in CPU
            memory so that the block data is readily available for merging new
            blocks. Only one GPU may access 'cpu_copy' at once. The OpenGL textures
            are updated from cpu_copy whenever new_data_available is set to true.

            For single-GPU environments, 'cpu_copy' is not used.
        */
    #ifndef SAWE_NO_MUTEX
        pData cpu_copy;
        bool new_data_available;
        QMutex cpu_copy_mutex;
    #endif

        /**
          valid_samples describes the intervals of valid samples contained in this block.
          it is relative to the start of the heightmap, not relative to this block unless this is
          the first block in the heightmap. The samplerate is the sample rate of the full
          resolution signal.
          */
        Signal::Intervals valid_samples, non_zero;
    };
    typedef boost::shared_ptr<Block> pBlock;

} // namespace Heightmap

#endif // BLOCK_H
