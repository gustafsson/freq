#include "propertiesstroke.h"
#include "ui_propertiesstroke.h"

namespace Saweui {

PropertiesStroke::PropertiesStroke(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::PropertiesStroke)
{
    ui->setupUi(this);
}

PropertiesStroke::~PropertiesStroke()
{
    delete ui;
}

void PropertiesStroke::changeEvent(QEvent *e)
{
    QWidget::changeEvent(e);
    switch (e->type()) {
    case QEvent::LanguageChange:
        ui->retranslateUi(this);
        break;
    default:
        break;
    }
}

} // namespace Saweui
