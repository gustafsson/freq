#ifndef MATLABCONTROLLER_H
#define MATLABCONTROLLER_H

#include "adapters/readmatlabsettings.h"
#include "sawe/toolmodel.h"

#include <signal/operation.h>

#include <QObject>
#include <QPointer>

namespace Sawe { class Project; }
namespace Signal { class Worker; }
namespace Adapters { class MatlabOperation; }

class QMenu;
class QToolBar;

namespace Tools
{
    class MatlabOperationWidget;
    class RenderView;

    class MatlabController: public QObject {
        Q_OBJECT
#if !defined(TARGET_reader)
    public:
        MatlabController( Sawe::Project* project, RenderView* render_view );
        ~MatlabController();

    private slots:
        void receiveMatlabOperation();
        void receiveMatlabFilter();
        void tryHeadAsMatlabOperation();
        void createFromAction();
        void foundNewScript( Adapters::DefaultMatlabFunctionSettings settings );
        void createFromScriptPath();
        void showDialogFromSettings(Adapters::DefaultMatlabFunctionSettings settings);
        void sourceRead();
        void createFromSettingsFailed( QString filename, QString info );

    private:
        void showNewMatlabOperationDialog( Adapters::MatlabFunctionSettings* psettings );
        void createFromSettings( Adapters::MatlabFunctionSettings& settings );

        void createView();
        void createView(Signal::Operation* o);

        void updateScriptsMenu();
        void createOperation(MatlabOperationWidget* settings);
        void connectOperation(MatlabOperationWidget* settings, Signal::pOperation matlaboperation);
        void updateStoredSettings(Adapters::MatlabFunctionSettings* settings);

        // Model
        // Model that is controlled, this controller doesn't have a view
        // and shares control of the worker with many others
        Sawe::Project* project_;

        RenderView* render_view_;
        QPointer<QMenu> scripts_;
        QPointer<QToolBar> scriptsToolbar_;

        // GUI
        // The fact that this controller doesn't have a view doesn't mean
        // it can't have widgets. QToolButton widgets are created by adding
        // their corresponding action to a ToolBar.
        // TODO Could also add this functionality to a menu.
        void setupGui();
        void prepareLogView( Adapters::MatlabOperation*m );
#endif // TARGET_reader
    };
} // namespace Tools

#endif // MATLABCONTROLLER_H
