#include "sawe/csv.h"
#include <sstream>
#include <fstream>
#include "tfr/cwt.h"
#include "tfr/cwtchunk.h"

using namespace std;

namespace Sawe
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
        operator()( Tfr::Chunk& c )
{
    string filename = csv_number();
    TaskTimer tt("Saving CSV-file %s", filename.c_str());
    ofstream csv(filename.c_str());

    Tfr::Chunk* chunk;
    Tfr::pChunk pchunk;
    Tfr::CwtChunk* cwt = dynamic_cast<Tfr::CwtChunk*>(&c);

    if (cwt)
    {
        pchunk = cwt->cleanChunk();
        chunk = pchunk.get();
    }
    else
        chunk = &c;

    float2* p = chunk->transform_data->getCpuMemory();
    cudaExtent s = chunk->transform_data->getNumberOfElements();

    for (unsigned y = 0; y<s.height; y++) {
        stringstream ss;
        for (unsigned x = 0; x<s.width; x++) {
            float2& v = p[x + y*s.width];
            ss << v.x << " " << v.y << " ";
        }
        csv << ss.str() << endl;
    }
}

} // namespace Sawe
