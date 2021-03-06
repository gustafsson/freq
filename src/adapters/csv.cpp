#include "csv.h"
#include "tfr/cwt.h"
#include "tfr/cwtchunk.h"

#include "tasktimer.h"

#include <sstream>
#include <fstream>

using namespace std;

namespace Adapters
{

string csv_number()
{
    string basename = "sonicawe-";
    for (unsigned c = 1; c<10; c++)
    {
        stringstream filename;
        filename << basename << c << ".csv";
        fstream csv(filename.str().c_str());
        if (!csv.is_open())
            return filename.str();
    }
    return basename+"0.csv";
}


void Csv::
        operator()( Tfr::ChunkAndInverse& chunkai )
{
    Tfr::Chunk& c = *chunkai.chunk;
    string filename;
    if (this->_filename.empty())
        filename = csv_number();
    else
        filename = this->_filename;

    TaskTimer tt("Saving CSV-file %s", filename.c_str());
    ofstream csv(filename.c_str());

    Tfr::Chunk* chunk;
    Tfr::pChunk pchunk;
    Tfr::CwtChunkPart* cwt = dynamic_cast<Tfr::CwtChunkPart*>(&c);

    if (cwt)
    {
        pchunk = cwt->cleanChunk();
        chunk = pchunk.get();
    }
    else
        chunk = &c;

    std::complex<float>* p = chunk->transform_data->getCpuMemory();
    DataStorageSize s = chunk->transform_data->size();

    for (int y = 0; y<s.height; y++) {
        stringstream ss;
        for (int x = 0; x<s.width; x++) {
            std::complex<float>& v = p[x + y*s.width];
            ss << v.real() << " " << v.imag() << " ";
        }
        csv << ss.str() << endl;
    }
}


Tfr::pChunkFilter CsvDesc::
        createChunkFilter(Signal::ComputingEngine* engine) const
{
    if (engine==0 || dynamic_cast<Signal::ComputingCpu*>(engine))
        return Tfr::pChunkFilter(new Csv(filename_));
    return Tfr::pChunkFilter();
}


Tfr::ChunkFilterDesc::ptr CsvDesc::
        copy() const
{
    return ChunkFilterDesc::ptr( new CsvDesc(filename_));
}

} // namespace Adapters
