#ifndef TOOLBAR_H
#define TOOLBAR_H

#include <QToolBar>

namespace Tools { namespace Support {

class ToolBar : public QToolBar
{
    Q_OBJECT
public:
    explicit ToolBar(QWidget *parent = 0);

signals:
    void visibleChanged(bool visible);

public slots:

private:
    virtual void showEvent(QShowEvent*);
    virtual void hideEvent(QHideEvent*);
};

} // namespace Support
} // namespace Tools

#endif // TOOLBAR_H
