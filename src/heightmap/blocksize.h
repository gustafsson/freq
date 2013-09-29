#ifndef HEIGHTMAP_BLOCKCONFIGURATION_H
#define HEIGHTMAP_BLOCKCONFIGURATION_H

#include <string>

namespace Heightmap {

typedef float SampleRate;


/**
 * @brief The BlockSize class should describe the size in texels of blocks in
 * a heightmap.
 *
 * It should be immutable POD.
 */
class BlockSize {
public:
    BlockSize(int texels_per_row, int texels_per_column);

    int texels_per_row() const { return texels_per_row_; }
    int texels_per_column() const { return texels_per_column_; }
    int texels_per_block() const { return texels_per_row() * texels_per_column(); }

    bool operator==(const BlockSize& b);
    bool operator!=(const BlockSize& b);

    std::string toString() const;

    template< class ostream_t > inline
    friend ostream_t& operator<<(ostream_t& os, const BlockSize& r) {
        return os << r.toString();
    }

private:
    int texels_per_column_;
    int texels_per_row_;

public:
    static void test();
};


/**
 * @brief The BlockLayout class should describe the sizes of blocks.
 *
 * It should be immutable POD.
 */
class BlockLayout {
public:
    BlockLayout(BlockSize, SampleRate fs);

    BlockSize               block_size() const;
    SampleRate              targetSampleRate() const;

    bool operator==(const BlockLayout& b);
    bool operator!=(const BlockLayout& b);

private:
    BlockSize               block_size_;
    SampleRate              targetSampleRate_;

public:
    static void test();
};


} // namespace Heightmap

#endif // HEIGHTMAP_BLOCKCONFIGURATION_H
