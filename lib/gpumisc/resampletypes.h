#ifndef RESAMPLETYPES_H
#define RESAMPLETYPES_H

#ifndef RESAMPLE_CALL
#define RESAMPLE_CALL
#endif

#ifndef RESAMPLE_ANYCALL
#define RESAMPLE_ANYCALL RESAMPLE_CALL
#endif


struct DataPos
{
    RESAMPLE_ANYCALL DataPos(int x, int y=0) : x(x), y(y) {}

    int x, y;
};


struct ValidSamples
{
    ValidSamples()
        : left(0), top(0), right(0), bottom(0) {}

    ValidSamples(float left, float top, float right, float bottom)
        : left(left), top(top), right(right), bottom(bottom) {}

    RESAMPLE_ANYCALL float width() { return right-left; }
    RESAMPLE_ANYCALL float height() { return bottom-top; }

    int left, top, right, bottom;
};


struct ResamplePos
{
    RESAMPLE_ANYCALL ResamplePos() : x(0), y(0) {}
    RESAMPLE_ANYCALL ResamplePos(float x, float y) : x(x), y(y) {}

    float x, y;
};


struct ResampleArea
{
    RESAMPLE_ANYCALL ResampleArea()
        : left(0), top(0), right(0), bottom(0) {}

    RESAMPLE_ANYCALL ResampleArea( const ValidSamples& v )
        : left(v.left), top(v.top), right(v.right), bottom(v.bottom) {}

    RESAMPLE_ANYCALL ResampleArea(float left, float top, float right, float bottom)
        : left(left), top(top), right(right), bottom(bottom) {}

    RESAMPLE_ANYCALL float width() { return right-left; }
    RESAMPLE_ANYCALL float height() { return bottom-top; }

    float left, top, right, bottom;
};

#endif // RESAMPLETYPES_H
