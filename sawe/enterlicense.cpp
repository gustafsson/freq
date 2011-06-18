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
    if (!tryread(ui->lineEdit->text().toStdString()).empty())
        accept();
}

} // namespace Sawe
