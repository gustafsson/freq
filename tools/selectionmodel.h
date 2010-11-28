#ifndef SELECTIONMODEL_H
#define SELECTIONMODEL_H

namespace Sawe {
    class Project;
}

/*#include "signal/postsink.h"
#include "signal/worker.h"
*/
#include "signal/operation.h"
#include <vector>


namespace Tools
{
    // TODO SelectionModel skall bara beskriva en selection
    class SelectionModel
    {
    public:
        SelectionModel(Sawe::Project* p);
        ~SelectionModel();


        Sawe::Project* project;

        Signal::pOperation current_filter_;
        std::vector<Signal::pOperation> all_filters;
    };
} // namespace Tools

#endif // SELECTIONMODEL_H
