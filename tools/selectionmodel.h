#ifndef SELECTIONMODEL_H
#define SELECTIONMODEL_H

#include "sawe/project.h"

struct MyVector { // TODO use gpumisc/tvector
    float x, y, z;
};

namespace Tools
{
    class SelectionModel
    {
    public:
        SelectionModel(Sawe::Project* p) : project(p) {}

        Signal::pWorkerCallback postsinkCallback;

        MyVector selection[2];

        Sawe::Project* project;
    };
} // namespace Tools

#endif // SELECTIONMODEL_H
