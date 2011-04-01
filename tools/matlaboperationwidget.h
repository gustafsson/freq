#ifndef MATLABOPERATIONWIDGET_H
#define MATLABOPERATIONWIDGET_H

#include "signal/operation.h"
#include "signal/target.h"

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
    explicit MatlabOperationWidget(Sawe::Project* project, QWidget *parent = 0);
    ~MatlabOperationWidget();

    std::string scriptname();
    void scriptname(std::string);

    std::string arguments();
    void arguments(std::string);

    virtual int chunksize();
    void chunksize(int);

    virtual bool computeInOrder();
    void computeInOrder(bool);

    virtual int redundant();
    virtual void redundant(int);

    Signal::pOperation ownOperation;

    void setOperation( Signal::pOperation om );
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
    void postRestartScript();
    void chunkSizeChanged();
    void restoreChanges();

    void sendCommand();

    void finished ( int exitCode, QProcess::ExitStatus exitStatus );

    void checkOctaveVisibility();

private:
    QPointer<QProcess> pid;
    void setProcess(QProcess*);
    virtual void hideEvent ( QHideEvent * event );

    Ui::MatlabOperationWidget *ui;

    Adapters::DefaultMatlabFunctionSettings prevsettings;

    Signal::pChain matlabChain;
    Signal::pTarget matlabTarget;

    Sawe::Project* project;
    QPointer<QDockWidget> octaveWindow;
    QPlainTextEdit* text;
    QVBoxLayout* verticalLayout;
    QLineEdit* edit;
    QTimer announceInvalidSamplesTimer;
};


} // namespace Tools
#endif // MATLABOPERATIONWIDGET_H
