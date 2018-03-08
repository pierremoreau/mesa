//
// Copyright 2017 Pierre Moreau
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
// THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
// OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
// ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
//

#ifndef CLOVER_SPIRV_INVOCATION_HPP
#define CLOVER_SPIRV_INVOCATION_HPP

#include "core/module.hpp"
#include "core/program.hpp"

namespace clover {
   namespace spirv {
      // Returns true if the binary starts with the SPIR-V magic word, false
      // otherwise.
      //
      // The first word is interpreted as little endian and big endian, but
      // only one of them has to match.
      bool is_binary_spirv(const char *binary);

      // Creates a clover module out of the given SPIR-V binary.
      module process_program(const std::vector<char> &binary,
                             const device &dev, bool validate,
                             std::string &r_log);

      // Combines multiple clover modules into a single one, resolving
      // link dependencies between them.
      module link_program(const std::vector<module> &modules, const device &dev,
                          const std::string &opts, std::string &r_log);

      bool is_valid_spirv(const uint32_t *binary, size_t length,
                          const context::notify_action &notify);
   }
}

#endif
