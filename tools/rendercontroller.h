#ifndef RENDERCONTROLLER_H
#define RENDERCONTROLLER_H

#include "renderview.h"
#include <QWidget>

namespace Tools
{
    class RenderController:public QWidget
    {
        Q_OBJECT
    public:
        RenderController( RenderView *view );
        ~RenderController();

    private:
        RenderModel *model;
        RenderView *view;
    };
} // namespace Tools

#endif // RENDERCONTROLLER_H
