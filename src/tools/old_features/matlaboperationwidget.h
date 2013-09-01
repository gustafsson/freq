#if 0
#ifndef MATLABOPERATIONWIDGET_H
#define MATLABOPERATIONWIDGET_H

#include "signal/operation.h"

#include "adapters/matlaboperation.h"

#include <QWidget>
#include <QTimer>
#include <QProcess>
#include <QPointer>

class QDockWidget;
class QPlainTextEdit;
class QLineEdit;
class QVBoxLayout;

namespace Sawe { class Project; }

namespace Tools {

namespace Ui {
    class MatlabOperationWidget;
}

class MatlabOperationWidget : public QWidget, public Adapters::MatlabFunctionSettings
{
    Q_OBJECT

public:
    explicit MatlabOperationWidget(Adapters::MatlabFunctionSettings* settings, Sawe::Project* project, QWidget *parent = 0);
    ~MatlabOperationWidget();

    virtual std::string scriptname() const;
    void scriptname(const std::string&);

    virtual std::string arguments() const;
    void arguments(const std::string&);

    virtual std::string argument_description() const;
    virtual void argument_description(const std::string&);

    virtual int chunksize() const;
    void chunksize(int);

    virtual bool computeInOrder() const;
    void computeInOrder(bool);

    virtual int overlap() const;
    virtual void overlap(int);

    /**
      For console 'ownOperation' is the only instance of a shared_ptr to the MatlabOperation
      that communicates with Matlab/Octave.

      For scripts (common usage) 'ownOperation' is the composite operation that is added
      to the tree of operations. Including all cache layers and including any filtering.
      */
    Signal::pOperation ownOperation;

    QDockWidget* getOctaveWindow();

    bool hasProcess();

public slots:
    void showOutput();

private slots:
    void browse();

    void populateTodoList();
    void announceInvalidSamples();
    void invalidateAllSamples();
    void restartScript();
    void reloadAutoSettings();
    void settingsRead( Adapters::DefaultMatlabFunctionSettings settings );
    void postRestartScript();
    void chunkSizeChanged();
    void restoreChanges();
    void settingsVisibleToggled(bool);

    void sendCommand();

    void finished ( int exitCode, QProcess::ExitStatus exitStatus );

    void checkOctaveVisibility();

private:
    bool hasValidTarget();

    QPointer<QProcess> pid;
    void setProcess(QProcess*);
    virtual void hideEvent ( QHideEvent * event );

    Ui::MatlabOperationWidget *ui;

    Adapters::DefaultMatlabFunctionSettings prevsettings;

//    Signal::pChain matlabChain;
//    Signal::pTarget matlabTarget;

    Sawe::Project* project;
    QPointer<QDockWidget> octaveWindow;
    QPlainTextEdit* text;
    QVBoxLayout* verticalLayout;
    QLineEdit* edit;
    QTimer announceInvalidSamplesTimer;
    bool hasCrashed;
};


} // namespace Tools
#endif // MATLABOPERATIONWIDGET_H
#endif
