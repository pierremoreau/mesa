//
// Copyright 2012 Francisco Jerez
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

#include "core/program.hpp"
#include "llvm/invocation.hpp"
#include "spirv/invocation.hpp"

using namespace clover;

namespace {
   module
   compile_program(const program &prog, const device &dev,
                   const std::string &opts, const header_map &headers,
                   std::string &log) {
      if (dev.ir_format() == PIPE_SHADER_IR_SPIRV) {
#ifdef CLOVER_ALLOW_SPIRV
         if (!prog.source().empty())
            return llvm::compile_to_spirv(prog.source(), headers, dev, opts,
                                          log);
         else if (prog.il_type() == program::il_type::spirv)
            return spirv::process_program(prog.il(), dev, false, log);
         else
#endif
            throw error(CL_INVALID_VALUE);
      } else {
         if (!prog.source().empty())
            return llvm::compile_program(prog.source(), headers, dev, opts, log);
#ifdef CLOVER_ALLOW_SPIRV
         else if (prog.il_type() == program::il_type::spirv)
            return llvm::compile_from_spirv(prog.il(), dev, log);
#endif
         else
            throw error(CL_INVALID_VALUE);
      }
   }
} // end of anonymous namespace

program::program(clover::context &ctx, const std::string &source) :
   has_source(true), has_il(false), context(ctx), _devices(ctx.devices()),
   _source(source), _kernel_ref_counter(0), _il(), _il_type(il_type::none) {
}

program::program(clover::context &ctx,
                 const ref_vector<device> &devs,
                 const std::vector<module> &binaries) :
   has_source(false), has_il(false), context(ctx), _devices(devs),
   _kernel_ref_counter(0), _il(), _il_type(il_type::none) {
   for_each([&](device &dev, const module &bin) {
         _builds[&dev] = { bin };
      },
      devs, binaries);
}

program::program(clover::context &ctx, const char *il, size_t length,
                 enum il_type il_type) :
   has_source(false), has_il(true), context(ctx), _devices(ctx.devices()),
   _kernel_ref_counter(0), _il(il, il + length), _il_type(il_type) {
}

void
program::compile(const ref_vector<device> &devs, const std::string &opts,
                 const header_map &headers) {
   if (has_source || has_il) {
      _devices = devs;

      for (auto &dev : devs) {
         std::string log;

         try {
            assert(dev.ir_format() == PIPE_SHADER_IR_NATIVE ||
                   dev.ir_format() == PIPE_SHADER_IR_SPIRV);
            _builds[&dev] = { compile_program(*this, dev, opts, headers, log),
               opts, log };
         } catch (...) {
            _builds[&dev] = { module(), opts, log };
            throw;
         }
      }
   }
}

void
program::link(const ref_vector<device> &devs, const std::string &opts,
              const ref_vector<program> &progs) {
   _devices = devs;

   for (auto &dev : devs) {
      const std::vector<module> ms = map([&](const program &prog) {
         return prog.build(dev).binary;
         }, progs);
      std::string log = _builds[&dev].log;

      try {
         assert(dev.ir_format() == PIPE_SHADER_IR_NATIVE ||
                dev.ir_format() == PIPE_SHADER_IR_SPIRV);
         const module m = dev.ir_format() == PIPE_SHADER_IR_SPIRV ?
            spirv::link_program(ms, dev, opts, log) :
            llvm::link_program(ms, dev, opts, log);
         _builds[&dev] = { m, opts, log };
      } catch (...) {
         _builds[&dev] = { module(), opts, log };
         throw;
      }
   }
}

const std::vector<char> &
program::il() const {
   return _il;
}

enum program::il_type
program::il_type() const {
   return _il_type;
}

const std::string &
program::source() const {
   return _source;
}

program::device_range
program::devices() const {
   return map(evals(), _devices);
}

cl_build_status
program::build::status() const {
   if (!binary.secs.empty())
      return CL_BUILD_SUCCESS;
   else if (log.size())
      return CL_BUILD_ERROR;
   else
      return CL_BUILD_NONE;
}

cl_program_binary_type
program::build::binary_type() const {
   if (any_of(type_equals(module::section::text_intermediate), binary.secs))
      return CL_PROGRAM_BINARY_TYPE_COMPILED_OBJECT;
   else if (any_of(type_equals(module::section::text_library), binary.secs))
      return CL_PROGRAM_BINARY_TYPE_LIBRARY;
   else if (any_of(type_equals(module::section::text_executable), binary.secs))
      return CL_PROGRAM_BINARY_TYPE_EXECUTABLE;
   else
      return CL_PROGRAM_BINARY_TYPE_NONE;
}

const struct program::build &
program::build(const device &dev) const {
   static const struct build null;
   return _builds.count(&dev) ? _builds.find(&dev)->second : null;
}

const std::vector<module::symbol> &
program::symbols() const {
   if (_builds.empty())
      throw error(CL_INVALID_PROGRAM_EXECUTABLE);

   return _builds.begin()->second.binary.syms;
}

unsigned
program::kernel_ref_count() const {
   return _kernel_ref_counter.ref_count();
}
