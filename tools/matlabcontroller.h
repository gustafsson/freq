#ifndef MATLABCONTROLLER_H
#define MATLABCONTROLLER_H

#include <signal/operation.h>

#include <QObject>

namespace Sawe { class Project; }
namespace Signal { class Worker; }

namespace Tools
{
    class RenderView;

    class MatlabController: public QObject {
        Q_OBJECT
    public:
        MatlabController( Sawe::Project* project, RenderView* render_view );

    private slots:
        virtual void receiveMatlabOperation(bool);
        virtual void receiveMatlabFilter(bool);

    private:
        // Model
        // Model that is controlled, this controller doesn't have a view
        // and shares control of the worker with many others
        Signal::Worker* _model;

        RenderView* render_view_;

        // GUI
        // The fact that this controller doesn't have a view doesn't mean
        // it can't have widgets. QToolButton widgets are created by adding
        // their corresponding action to a ToolBar.
        // TODO Could also add this functionality to a menu.
        void setupGui(Sawe::Project* project);

        // state
        // There is only one matlab filter and one matlab operation per
        // project instance. Becuase we havn't implemented any user interface
        // for distinguishing between different matlab scripts.
        Signal::pOperation _matlabfilter;
        Signal::pOperation _matlaboperation;
    };
} // namespace Tools
#endif // MATLABCONTROLLER_H
