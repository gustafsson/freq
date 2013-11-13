#ifndef ENTERLICENSE_H
#define ENTERLICENSE_H

#include <QDialog>

namespace Ui {
    class EnterLicense;
}

namespace Sawe
{

class EnterLicense : public QDialog
{
    Q_OBJECT

public:
    explicit EnterLicense();
    ~EnterLicense();

    QString lineEdit();

private slots:
    void validate();

private:
    Ui::EnterLicense *ui;
};

} // namespace Sawe

#endif // ENTERLICENSE_H
