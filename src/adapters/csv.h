#ifndef ADAPTERS_CSV_H
#define ADAPTERS_CSV_H

#include <string>

#include "tfr/cwtfilter.h"

namespace Adapters {

/**
  Transforms a pBuffer into a pChunk with Cwt and saves the chunk in a file called
  sonicawe-x.csv, where x is a number between 1 and 9, or 0 if all the other 9 files already
  exists. The file is saved with the csv-format comma separated values, but values are
  actually separated by spaces. One row of the csv-file corresponds to one row of the chunk.
*/
class Csv: public Tfr::CwtFilter
{
public:
    Csv(std::string filename="") : _filename(filename) {}

    virtual bool operator()( Tfr::Chunk& );

private:
    std::string _filename;
};

} // namespace Adapters

#endif // ADAPTERS_CSV_H
