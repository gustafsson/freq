#ifndef EXPORTAUDIODIALOG_H
#define EXPORTAUDIODIALOG_H

#include <QDialog>
#include <QTimer>

#include "selectionmodel.h"
#include "signal/worker.h"

namespace Sawe { class Project; }
namespace Ui { class ExportAudioDialog; }

namespace Tools {
class RenderView;

class ExportAudioDialog : public QDialog
{
    Q_OBJECT

public:
    ExportAudioDialog(
            Sawe::Project* project,
            SelectionModel* selection_model,
            RenderView* render_view );
    ~ExportAudioDialog();

public slots:
    void exportEntireFile();
    void exportSelection();
    void abortExport();
    void selectionChanged();
    void populateTodoList();
    void checkboxToggled(bool);
    void dialogFinished(int);
    void callUpdate();

private:
    Ui::ExportAudioDialog *ui;

    Sawe::Project* project;
    SelectionModel* selection_model;
    RenderView* render_view;

    Signal::pTarget exportTarget;
    QTimer update_timer;
    QString filemame;
    Signal::IntervalType total;
    bool drawnFinished_;;

    // overloaded from QWidget
    virtual void paintEvent ( QPaintEvent * event );

    void setupGui();

    void start(Signal::pOperation filter);
};


} // namespace Tools
#endif // EXPORTAUDIODIALOG_H
