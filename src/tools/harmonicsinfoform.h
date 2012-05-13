#ifndef HARMONICSINFOFORM_H
#define HARMONICSINFOFORM_H

#include <QWidget>

class QDockWidget;

namespace Sawe { class Project; }
namespace Ui { class HarmonicsInfoForm; }

namespace Tools {

    class TooltipController;
    class TooltipView;
    class RenderView;

class HarmonicsInfoForm : public QWidget
{
    Q_OBJECT

public:
    HarmonicsInfoForm(Sawe::Project*project, TooltipController* tooltipcontroller, RenderView* render_view);
    virtual ~HarmonicsInfoForm();

private slots:
    void checkVisibility(bool visible);
    void harmonicsChanged();
    void currentCellChanged();
    void deleteCurrentRow();

private:
    Ui::HarmonicsInfoForm *ui;
    Sawe::Project* project;
    TooltipController* harmonicscontroller;
    RenderView* render_view;

    QAction* deleteRow;
    QAction* actionHarmonics_info;
    QDockWidget* dock;

    int rebuilding;

    void addRow(TooltipView* view);
    void setCellInLastRow(int column, QString text);
};


} // namespace Tools
#endif // HARMONICSINFOFORM_H
