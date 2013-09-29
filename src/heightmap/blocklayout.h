#ifndef HEIGHTMAP_BLOCKLAYOUT_H
#define HEIGHTMAP_BLOCKLAYOUT_H

#include <string>

namespace Heightmap {

typedef float SampleRate;


/**
 * @brief The BlockLayout class should describe the size in texels of blocks in
 * a heightmap.
 *
 * It should be immutable POD.
 */
class BlockLayout {
public:
    BlockLayout(int texels_per_row, int texels_per_column, SampleRate fs);

    int texels_per_row() const { return texels_per_row_; }
    int texels_per_column() const { return texels_per_column_; }
    int texels_per_block() const { return texels_per_row() * texels_per_column(); }

    // sample rate is used to compute which rawdata (Signal::Interval) that a block represents
    SampleRate sample_rate() const { return sample_rate_; }
    SampleRate targetSampleRate() const { return sample_rate_; }

    bool operator==(const BlockLayout& b);
    bool operator!=(const BlockLayout& b);

    std::string toString() const;

    template< class ostream_t > inline
    friend ostream_t& operator<<(ostream_t& os, const BlockLayout& r) {
        return os << r.toString();
    }

private:
    int texels_per_column_;
    int texels_per_row_;
    SampleRate sample_rate_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKLAYOUT_H
