#ifndef SAWEHDF5_H
#define SAWEHDF5_H

#include "tfr-chunksink.h"

namespace Sawe {

/**
  Transforms a pBuffer into a pChunk with CwtSingleton and saves the chunk in a file called
  sonicawe-x.csv, where x is a number between 1 and 9, or 0 if all the other 9 files already
  exists. The file is saved with the csv-format comma separated values, but values are
  actually separated by spaces. One row of the csv-file corresponds to one row of the chunk.
*/
class Hdf5: public Tfr::ChunkSink
{
public:
    enum DataType {
        DataType_CHUNK,
        DataType_BUFFER
    };

    Hdf5(std::string filename="sawe_chunk.h5", bool saveChunk=true);

    void    put( Signal::pBuffer , Signal::pSource );

    static void             saveBuffer( std::string filename, const Signal::Buffer& );
    static void             saveChunk( std::string filename, const Tfr::Chunk& );

    static Signal::pBuffer  loadBuffer( std::string filename );
    static Tfr::pChunk      loadChunk( std::string filename );

private:

    bool _saveChunk;
    std::string _filename;
};

} // namespace Sawe

#endif // SAWEHDF5_H
