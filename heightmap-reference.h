#ifndef HEIGHTMAPCOLLECTION_H
#define HEIGHTMAPCOLLECTION_H

namespace Heightmap {


class Reference {
public:
    tvector<2,int> log2_samples_size;
    tvector<2,unsigned> block_index;

    bool operator==(const Reference &b) const;
    void getArea( Position &a, Position &b) const;
    unsigned sampleOffset() const;
    unsigned scaleOffset() const;

    bool containsSpectrogram() const;
    bool toLarge() const;

    /* child references */
    Reference left();
    Reference right();
    Reference top();
    Reference bottom();

    /* sibblings, 3 other references who share the same parent */
    Reference sibbling1();
    Reference sibbling2();
    Reference sibbling3();

    /* parent */
    Reference parent();
private:
    friend class Spectrogram;

    Reference( Spectrogram* parent );

    Spectrogram* _spectrogram;
};

} // namespace Heightmap

#endif // HEIGHTMAPCOLLECTION_H
