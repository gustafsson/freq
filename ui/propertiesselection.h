#ifndef PROPERTIESSELECTION_H
#define PROPERTIESSELECTION_H

#include <QWidget>

namespace Ui {

namespace Ui {
    class PropertiesSelection;
}

class PropertiesSelection : public QWidget {
    Q_OBJECT
public:
    PropertiesSelection(QWidget *parent = 0);
    ~PropertiesSelection();

protected:
    void changeEvent(QEvent *e);

private:
    Ui::PropertiesSelection *ui;
};


} // namespace Ui
#endif // PROPERTIESSELECTION_H
