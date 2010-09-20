#ifndef SELECTIONMODEL_H
#define SELECTIONMODEL_H

namespace Sawe {
    class Project;
}

#include "signal/postsink.h"
#include "signal/worker.h"

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

        Sawe::Project* project;
    };
} // namespace Tools

#endif // SELECTIONMODEL_H
