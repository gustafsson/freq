#ifndef PROPERTIESSELECTION_H
#define PROPERTIESSELECTION_H

#include <QWidget>

namespace Saweui {
namespace Ui {
    class PropertiesSelection;
}
}
namespace Ui {


class PropertiesSelection : public QWidget {
    Q_OBJECT
public:
    PropertiesSelection(QWidget *parent = 0);
    ~PropertiesSelection();

protected:
    void changeEvent(QEvent *e);

private:
    Saweui::Ui::PropertiesSelection *ui;
};


} // namespace Ui
#endif // PROPERTIESSELECTION_H
