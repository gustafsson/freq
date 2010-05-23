#ifndef SIGNALWRITEWAV_H
#define SIGNALWRITEWAV_H

#include "signal-worker.h"
#include "signal-samplesintervaldescriptor.h"
#include <vector>

namespace Signal {

class WriteWav: virtual public Sink
{
public:
    WriteWav( std::string filename );
    ~WriteWav();

    virtual void put( pBuffer );
    virtual void reset();

    SamplesIntervalDescriptor getMissingSamples();

private:
    std::string _filename;
    std::vector<pBuffer> _cache;

    unsigned nAccumulatedSamples();
    void writeToDisk();
    void writeToDisk(pBuffer);
};

} // namespace Signal

#endif // SIGNALWRITEWAV_H
