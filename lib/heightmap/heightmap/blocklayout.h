#ifndef HEIGHTMAP_BLOCKLAYOUT_H
#define HEIGHTMAP_BLOCKLAYOUT_H

#include <string>
#include <ctgmath>

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
    /**
     * @brief BlockLayout
     * @param block_width The width of the block texture in texels.
     * @param block_height The height of the block texture in texels.
     * @param fs The sample rate of the raw data. For mapping to Signal::Interval.
     * @param mipmaps How many mipmap levels that are used for increasing
     * contrast. Higher mipmap levels causes the overlapping margin to
     * increase. If no mipmap levels are used (mipmaps=0) the margin is 0.5
     * texels on each side so that adjacent blocks are rendered seamless.
     */
    BlockLayout(int texels_per_row, int texels_per_column, SampleRate fs, int mipmaps=0);

    int texels_per_row() const { return texels_per_row_; }
    int texels_per_column() const { return texels_per_column_; }
    int texels_per_block() const { return texels_per_row() * texels_per_column(); }
    int mipmaps() const { return mipmaps_; }
    float margin() const { return ldexpf(0.5f, mipmaps_); }
    int visible_texels_per_row() const { return texels_per_row_ - (1<<mipmaps_); }
    int visible_texels_per_column() const { return texels_per_column_ - (1<<mipmaps_); }

    // sample rate is used to compute which rawdata (Signal::Interval) that a block represents
    SampleRate sample_rate() const { return sample_rate_; }
    SampleRate targetSampleRate() const { return sample_rate_; }

    bool operator==(const BlockLayout& b) const;
    bool operator!=(const BlockLayout& b) const;

    std::string toString() const;

    template< class ostream_t > inline
    friend ostream_t& operator<<(ostream_t& os, const BlockLayout& r) {
        return os << r.toString();
    }

private:
    int texels_per_column_;
    int texels_per_row_;
    SampleRate sample_rate_;
    int mipmaps_;

public:
    static void test();
};

} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKLAYOUT_H
