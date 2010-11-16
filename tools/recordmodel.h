#ifndef RECORDMODEL_H
#define RECORDMODEL_H

#include <signal/operation.h>

namespace Sawe { class Project; }
namespace Adapters { class MicrophoneRecorder; }

namespace Tools
{

class RecordModel
{
public:
    RecordModel( Sawe::Project* project );
    ~RecordModel();

    static bool canCreateRecordModel( Sawe::Project* project );


    Adapters::MicrophoneRecorder* recording;
    Sawe::Project* project;
};

} // namespace Tools

#endif // RECORDMODEL_H
