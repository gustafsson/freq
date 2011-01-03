#ifndef REASSIGNCONTROLLER_H
#define REASSIGNCONTROLLER_H

#include <QObject>

namespace Sawe { class Project; }
namespace Signal { class Worker; }

namespace Tools
{
    class Reassign: public QObject {
        Q_OBJECT
    public:
        Reassign( Sawe::Project* project );

    private slots:
        virtual void receiveTonalizeFilter(bool);
        virtual void receiveReassignFilter(bool);

    private:
        // Model
        // Model that is controlled, this controller doesn't have a view
        // and shares control of the worker with many others
        Signal::Worker* _model;

        // GUI
        // The fact that this controller doesn't have a view doesn't mean
        // it can't have widgets. QToolButton widgets are created by adding
        // their corresponding action to a ToolBar.
        // TODO Could also add this functionality to a menu.
        void setupGui(Sawe::Project* project);
    };
} // namespace Tools
#endif // REASSIGNCONTROLLER_H
