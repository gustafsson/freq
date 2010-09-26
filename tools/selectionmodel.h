#ifndef SELECTIONMODEL_H
#define SELECTIONMODEL_H

namespace Sawe {
    class Project;
}

#include "signal/postsink.h"
#include "signal/worker.h"

#include <vector>

struct MyVector { // TODO use gpumisc/tvector
    float x, y, z;
};

namespace Tools
{
    class SelectionModel
    {
    public:
        SelectionModel(Sawe::Project* p);

        Signal::PostSink* getPostSink();
        Signal::pWorkerCallback postsinkCallback;

        Signal::pOperation filter;

        MyVector selection[2];

        // Tool move selection is not a method of the selection tool
        // TODO move to its own tool
        MyVector sourceSelection[2];

        Sawe::Project* project;

        std::vector<Signal::pOperation> all_filters;
    };
} // namespace Tools

#endif // SELECTIONMODEL_H
