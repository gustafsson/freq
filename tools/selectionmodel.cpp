#include "selectionmodel.h"
#include "sawe/project.h"

namespace Tools
{

SelectionModel::
        SelectionModel(Sawe::Project* p)
            : project(p)
{
}


SelectionModel::
        ~SelectionModel()
{
    TaskTimer(__FUNCTION__).suppressTiming();
}


} // namespace Tools
