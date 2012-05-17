#ifndef ABOUTDIALOG_H
#define ABOUTDIALOG_H

#include <QDialog>

namespace Ui {
    class AboutDialog;
}

namespace Sawe
{
    class Project;
}

namespace Tools
{

class AboutDialog : public QDialog
{
    Q_OBJECT

public:
    explicit AboutDialog(Sawe::Project* p);
    ~AboutDialog();

private:
    virtual void showEvent(QShowEvent *);

    Ui::AboutDialog *ui;
};

} // namespace Tools

#endif // ABOUTDIALOG_H
