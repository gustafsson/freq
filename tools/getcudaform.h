#ifndef GETCUDAFORM_H
#define GETCUDAFORM_H

#include <QWidget>
#include <QUrl>
#include <QTimer>

class QVBoxLayout;

namespace Tools {

namespace Ui {
    class GetCudaForm;
}

class GetCudaForm : public QWidget
{
    Q_OBJECT

public:
    explicit GetCudaForm(QWidget *parent = 0);
    ~GetCudaForm();

public slots:
    void readMore();
    void ani();

private:
    Ui::GetCudaForm *ui;
    QTimer animate;
    QVBoxLayout* parentLayout;
    int spacing;

    static QUrl url;
};


} // namespace Tools
#endif // GETCUDAFORM_H
