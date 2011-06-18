#include "enterlicense.h"
#include "ui_enterlicense.h"

EnterLicense::
        EnterLicense(QWidget *parent)
            :
    QDialog(parent),
    ui(new Ui::EnterLicense)
{
    ui->setupUi(this);
}

EnterLicense::
        ~EnterLicense()
{
    delete ui;
}

QString EnterLicense::
        lineEdit()
{
    return ui->lineEdit->text();
}
