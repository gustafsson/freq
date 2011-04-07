#ifndef REASSIGNCONTROLLER_H
#define REASSIGNCONTROLLER_H

#include <QObject>
#include "signal/chain.h"

namespace Sawe { class Project; }
namespace Signal { class Worker; }

namespace Tools
{
    class Reassign: public QObject {
        Q_OBJECT
    public:
        Reassign( Sawe::Project* project );

    private slots:
        virtual void receiveTonalizeFilter();
        virtual void receiveReassignFilter();

    private:
        // Model
        // Model that is controlled, this controller doesn't have a view
        // and shares control of the project current chainhead with many others
        Sawe::Project* project_;

        // GUI
        // The fact that this controller doesn't have a view doesn't mean
        // it can't have widgets. QToolButton widgets are created by adding
        // their corresponding action to a ToolBar.
        // TODO Could also add this functionality to a menu.
        void setupGui();
    };
} // namespace Tools
#endif // REASSIGNCONTROLLER_H
