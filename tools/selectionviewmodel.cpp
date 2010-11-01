// TODO remove? or merge with selectionview to adjust selection parameters

#include "selectionviewmodel.h"
#include "ui_selectionviewmodel.h"

SelectionViewModel::SelectionViewModel(QWidget *parent) :
    QDockWidget(parent),
    ui(new Ui::SelectionViewModel)
{
    ui->setupUi(this);
}

SelectionViewModel::~SelectionViewModel()
{
    delete ui;
}
