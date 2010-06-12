#ifndef PROPERTIESSELECTION_H
#define PROPERTIESSELECTION_H

#include <QWidget>

namespace Saweui {

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


} // namespace Saweui
#endif // PROPERTIESSELECTION_H
