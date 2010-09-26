// TODO remove
#ifndef SELECTIONVIEWMODEL_H
#define SELECTIONVIEWMODEL_H

#include <QDockWidget>

namespace Ui {
    class SelectionViewModel;
}

class SelectionViewModel : public QDockWidget
{
    Q_OBJECT

public:
    explicit SelectionViewModel(QWidget *parent = 0);
    ~SelectionViewModel();

private:
    Ui::SelectionViewModel *ui;
};

#endif // SELECTIONVIEWMODEL_H
