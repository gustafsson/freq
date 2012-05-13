#ifndef RECORDMODEL_H
#define RECORDMODEL_H

#include <signal/operation.h>

namespace Sawe { class Project; }
namespace Adapters { class Recorder; }

namespace Tools
{

class RenderView;

class RecordModel
{
public:
    RecordModel( Sawe::Project* project, RenderView* render_view );
    ~RecordModel();

    static bool canCreateRecordModel( Sawe::Project* project );


    Adapters::Recorder* recording;
    Sawe::Project* project;
    RenderView* render_view;
};

} // namespace Tools

#endif // RECORDMODEL_H
