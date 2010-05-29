#ifndef SIGNALWRITEWAV_H
#define SIGNALWRITEWAV_H

#include "signal-worker.h"
#include "signal-samplesintervaldescriptor.h"
#include "signal-sinksource.h"
#include <vector>

namespace Signal {

class WriteWav: public SinkSource
{
public:
    WriteWav( std::string filename );
    ~WriteWav();

    virtual void put( pBuffer );
    virtual void put( pBuffer b, pSource ) { put (b); }
    virtual void reset();
    virtual bool finished();

private:
    std::string _filename;

    void writeToDisk();
    void writeToDisk(pBuffer);
};

} // namespace Signal

#endif // SIGNALWRITEWAV_H
