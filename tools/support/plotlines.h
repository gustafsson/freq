#ifndef PLOTLINES_H
#define PLOTLINES_H

#include <QObject>

#include <map>
#include <vector>

#include "signal/intervals.h"

namespace Tools {
    class RenderModel;

namespace Support {

class PlotLines : public QObject
{
    Q_OBJECT
public:
    explicit PlotLines(RenderModel* render_model);

    struct Value {
        Value():hz(0), a(0) {}
        Value(float hz, float a):hz(hz),a(a) {}

        float hz;
        float a;
    };

    typedef float Time;

    struct Line
    {
        float R, G, B, A;
        // map is sorted on key=Time
        typedef std::map<Time, Value> Data;
        Data data;
    };

    typedef unsigned LineIdentifier;

    typedef std::map<LineIdentifier, Line> Lines;
    const Lines& lines() const { return lines_; }

    void clear();
    void clear( const Signal::Intervals& I, float fs );
    void set( Time t, float hz, float a = 1);
    void set( LineIdentifier id, Time t, float hz, float a = 1);

    Line& line(LineIdentifier id);

signals:

public slots:
    void draw();

private:
    Lines lines_;
    RenderModel* render_model_;
    float rand_color_offs_;

    void draw(Line& l);
    void recomputeColors();
};

} // namespace Support
} // namespace Tools

#endif // PLOTLINES_H
