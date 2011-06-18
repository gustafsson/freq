#ifndef ENTERLICENSE_H
#define ENTERLICENSE_H

#include <QDialog>

namespace Ui {
    class EnterLicense;
}

class EnterLicense : public QDialog
{
    Q_OBJECT

public:
    explicit EnterLicense(QWidget *parent = 0);
    ~EnterLicense();

    QString lineEdit();

private:
    Ui::EnterLicense *ui;
};

#endif // ENTERLICENSE_H
