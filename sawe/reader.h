#ifndef READER_H
#define READER_H

#include <string>
#include <vector>

namespace Sawe
{

class Reader
{
public:
    static std::string reader_text(bool annoy=false);
    static std::string reader_title();
    static std::string tryread(std::string mash);
};

} // namespace Sawe

#endif // READER_H
