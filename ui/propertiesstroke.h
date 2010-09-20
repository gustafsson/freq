#ifndef PROPERTIESSTROKE_H
#define PROPERTIESSTROKE_H

#include <QWidget>

namespace Saweui {
namespace Ui {
    class PropertiesStroke;
}
}

namespace Ui {


class PropertiesStroke : public QWidget {
    Q_OBJECT
public:
    PropertiesStroke(QWidget *parent = 0);
    ~PropertiesStroke();

protected:
    void changeEvent(QEvent *e);

private:
    Saweui::Ui::PropertiesStroke *ui;
};


} // namespace Ui
#endif // PROPERTIESSTROKE_H
