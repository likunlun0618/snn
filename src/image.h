#ifndef IMAGE_H
#define IMAGE_H

#include <string>
#include "tensor.h"

Tensor imread(std::string filename);

int imwrite(std::string filename, const Tensor &inp);

#endif // IMAGE_H
