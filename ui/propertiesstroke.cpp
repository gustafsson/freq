#include "propertiesstroke.h"
#include "ui_propertiesstroke.h"

namespace Ui {

PropertiesStroke::PropertiesStroke(QWidget *parent) :
    QWidget(parent),
    ui(new Saweui::Ui::PropertiesStroke)
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

} // namespace Ui
