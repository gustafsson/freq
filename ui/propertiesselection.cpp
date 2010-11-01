#include "propertiesselection.h"
#include "ui_propertiesselection.h"
#include <QToolBar>
#include <QToolButton>
#include <QLayout>

namespace Ui {

PropertiesSelection::PropertiesSelection(QWidget *parent) :
    QWidget(parent),
    ui(new Saweui::Ui::PropertiesSelection)
{
    ui->setupUi(this);

    /*QToolBar* tb = new QToolBar( this );
    tb->addAction( ui->actionActivateSelection );
    tb->addAction( ui->actionCutoffSelection );
    tb->addAction( ui->actionPeakSelection );*/
/*    QToolButton* tb = new QToolButton( this );
    this->setLayout( Qt::QLa Qt:: Qt::LayoutDirection);;)*/
}

PropertiesSelection::~PropertiesSelection()
{
    delete ui;
}

void PropertiesSelection::changeEvent(QEvent *e)
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
