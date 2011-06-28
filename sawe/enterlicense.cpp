#include "enterlicense.h"
#include "ui_enterlicense.h"

#include "reader.h"

namespace Sawe
{

EnterLicense::
        EnterLicense( )
            :
    QDialog(0),
    ui(new Ui::EnterLicense)
{
    ui->setupUi(this);
    QObject::disconnect(ui->buttonBox, SIGNAL(accepted()), this, SLOT(accept()));

    connect(ui->buttonBox, SIGNAL(accepted()), SLOT(validate()));
    connect(ui->lineEdit, SIGNAL(textChanged(QString)), SLOT(textChanged()));
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

void EnterLicense::
        validate()
{
    if (!Reader::tryread(ui->lineEdit->text().toStdString()).empty())
        accept();
    else
    {
        ui->label->setText("Your product key is not valid. Contact us at sonicawe.com to obtain a new key.");
    }
}

} // namespace Sawe
