/*
 * Copyright 2017 Red Hat Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and associated documentation files (the "Software"),
 * to deal in the Software without restriction, including without limitation
 * the rights to use, copy, modify, merge, publish, distribute, sublicense,
 * and/or sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
 * THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR
 * OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
 * ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 *
 * Authors: Karol Herbst <kherbst@redhat.com>
 */

#include "compiler/nir/nir.h"

#include "util/u_debug.h"

#include "codegen/nv50_ir.h"
#include "codegen/nv50_ir_from_common.h"
#include "codegen/nv50_ir_lowering_helper.h"
#include "codegen/nv50_ir_util.h"

#if __cplusplus >= 201103L
#include <unordered_map>
#else
#include <tr1/unordered_map>
#endif
#include <vector>

namespace {

#if __cplusplus >= 201103L
using std::hash;
using std::unordered_map;
#else
using std::tr1::hash;
using std::tr1::unordered_map;
#endif

using namespace nv50_ir;

int
type_size(const struct glsl_type *type)
{
   return glsl_count_attribute_slots(type, false);
}

class Converter : public ConverterCommon
{
public:
   Converter(Program *, nir_shader *, nv50_ir_prog_info *);

   bool run();
private:
   typedef std::vector<LValue *> LValues;
   typedef unordered_map<unsigned, LValues> NirDefMap;

   LValues& convert(nir_alu_dest *);
   LValues& convert(nir_dest *);
   LValues& convert(nir_register *);
   LValues& convert(nir_ssa_def *);

   Value* getSrc(nir_alu_src *, uint8_t component = 0);
   Value* getSrc(nir_register *, uint8_t);
   Value* getSrc(nir_src *, uint8_t, bool indirect = false);
   Value* getSrc(nir_ssa_def *, uint8_t);

   // returned value is the constant part of the given source (either the
   // nir_src or the selected source component of an intrinsic). Even though
   // this is mostly an optimization to be able to skip indirects in a few
   // cases, sometimes we require immediate values or set some fileds on
   // instructions (e.g. tex) in order for codegen to consume those.
   // If the found value has not a constant part, the Value gets returned
   // through the Value parameter.
   uint32_t getIndirect(nir_src *, uint8_t, Value *&);
   uint32_t getIndirect(nir_intrinsic_instr *, uint8_t s, uint8_t c, Value *&);

   bool isFloatType(nir_alu_type);
   bool isSignedType(nir_alu_type);
   bool isResultFloat(nir_op);
   bool isResultSigned(nir_op);

   DataType getDType(nir_alu_instr *);
   DataType getDType(nir_intrinsic_instr *);
   DataType getDType(nir_op, uint8_t);

   std::vector<DataType> getSTypes(nir_alu_instr *);
   DataType getSType(nir_src &, bool isFloat, bool isSigned);

   nir_shader *nir;

   NirDefMap ssaDefs;
   NirDefMap regDefs;
};

Converter::Converter(Program *prog, nir_shader *nir, nv50_ir_prog_info *info)
   : ConverterCommon(prog, info),
     nir(nir) {}

bool
Converter::isFloatType(nir_alu_type type)
{
   return nir_alu_type_get_base_type(type) == nir_type_float;
}

bool
Converter::isSignedType(nir_alu_type type)
{
   return nir_alu_type_get_base_type(type) == nir_type_int;
}

bool
Converter::isResultFloat(nir_op op)
{
   const nir_op_info &info = nir_op_infos[op];
   if (info.output_type != nir_type_invalid)
      return isFloatType(info.output_type);

   ERROR("isResultFloat not implemented for %s\n", nir_op_infos[op].name);
   assert(false);
   return true;
}

bool
Converter::isResultSigned(nir_op op)
{
   switch (op) {
   // there is no umul and we get wrong results if we treat all muls as signed
   case nir_op_imul:
   case nir_op_inot:
      return false;
   default:
      const nir_op_info &info = nir_op_infos[op];
      if (info.output_type != nir_type_invalid)
         return isSignedType(info.output_type);
      ERROR("isResultSigned not implemented for %s\n", nir_op_infos[op].name);
      assert(false);
      return true;
   }
}

DataType
Converter::getDType(nir_alu_instr *insn)
{
   if (insn->dest.dest.is_ssa)
      return getDType(insn->op, insn->dest.dest.ssa.bit_size);
   else
      return getDType(insn->op, insn->dest.dest.reg.reg->bit_size);
}

DataType
Converter::getDType(nir_intrinsic_instr *insn)
{
   if (insn->dest.is_ssa)
      return typeOfSize(insn->dest.ssa.bit_size / 8, false, false);
   else
      return typeOfSize(insn->dest.reg.reg->bit_size / 8, false, false);
}

DataType
Converter::getDType(nir_op op, uint8_t bitSize)
{
   DataType ty = typeOfSize(bitSize / 8, isResultFloat(op), isResultSigned(op));
   if (ty == TYPE_NONE) {
      ERROR("couldn't get Type for op %s with bitSize %u\n", nir_op_infos[op].name, bitSize);
      assert(false);
   }
   return ty;
}

std::vector<DataType>
Converter::getSTypes(nir_alu_instr *insn)
{
   const nir_op_info &info = nir_op_infos[insn->op];
   std::vector<DataType> res(info.num_inputs);

   for (uint8_t i = 0; i < info.num_inputs; ++i) {
      if (info.input_types[i] != nir_type_invalid) {
         res[i] = getSType(insn->src[i].src, isFloatType(info.input_types[i]), isSignedType(info.input_types[i]));
      } else {
         ERROR("getSType not implemented for %s idx %u\n", info.name, i);
         assert(false);
         res[i] = TYPE_NONE;
         break;
      }
   }

   return res;
}

DataType
Converter::getSType(nir_src &src, bool isFloat, bool isSigned)
{
   uint8_t bitSize;
   if (src.is_ssa)
      bitSize = src.ssa->bit_size;
   else
      bitSize = src.reg.reg->bit_size;

   DataType ty = typeOfSize(bitSize / 8, isFloat, isSigned);
   if (ty == TYPE_NONE) {
      const char *str;
      if (isFloat)
         str = "float";
      else if (isSigned)
         str = "int";
      else
         str = "uint";
      ERROR("couldn't get Type for %s with bitSize %u\n", str, bitSize);
      assert(false);
   }
   return ty;
}

Converter::LValues&
Converter::convert(nir_dest *dest)
{
   if (dest->is_ssa)
      return convert(&dest->ssa);
   if (dest->reg.indirect) {
      ERROR("no support for indirects.");
      assert(false);
   }
   return convert(dest->reg.reg);
}

Converter::LValues&
Converter::convert(nir_register *reg)
{
   NirDefMap::iterator it = regDefs.find(reg->index);
   if (it != regDefs.end())
      return it->second;

   LValues newDef(reg->num_components);
   for (uint8_t i = 0; i < reg->num_components; i++)
      newDef[i] = getScratch(std::max(4, reg->bit_size / 8));
   return regDefs[reg->index] = newDef;
}

Converter::LValues&
Converter::convert(nir_ssa_def *def)
{
   NirDefMap::iterator it = ssaDefs.find(def->index);
   if (it != ssaDefs.end())
      return it->second;

   LValues newDef(def->num_components);
   for (uint8_t i = 0; i < def->num_components; i++)
      newDef[i] = getSSA(std::max(4, def->bit_size / 8));
   return ssaDefs[def->index] = newDef;
}

Value*
Converter::getSrc(nir_alu_src *src, uint8_t component)
{
   if (src->abs || src->negate) {
      ERROR("modifiers currently not supported on nir_alu_src\n");
      assert(false);
   }
   return getSrc(&src->src, src->swizzle[component]);
}

Value*
Converter::getSrc(nir_register *reg, uint8_t idx)
{
   NirDefMap::iterator it = regDefs.find(reg->index);
   if (it == regDefs.end())
      return convert(reg)[idx];
   return it->second[idx];
}

Value*
Converter::getSrc(nir_src *src, uint8_t idx, bool indirect)
{
   if (src->is_ssa)
      return getSrc(src->ssa, idx);

   if (src->reg.indirect) {
      if (indirect)
         return getSrc(src->reg.indirect, idx);
      ERROR("no support for indirects.");
      assert(false);
      return NULL;
   }

   return getSrc(src->reg.reg, idx);
}

Value*
Converter::getSrc(nir_ssa_def *src, uint8_t idx)
{
   NirDefMap::iterator it = ssaDefs.find(src->index);
   if (it == ssaDefs.end()) {
      ERROR("SSA value %u not found\n", src->index);
      assert(false);
      return NULL;
   }
   return it->second[idx];
}

uint32_t
Converter::getIndirect(nir_src *src, uint8_t idx, Value *&indirect)
{
   nir_const_value *offset = nir_src_as_const_value(*src);

   if (offset) {
      indirect = NULL;
      return offset->u32[0];
   }

   indirect = getSrc(src, idx, true);
   return 0;
}

uint32_t
Converter::getIndirect(nir_intrinsic_instr *insn, uint8_t s, uint8_t c, Value *&indirect)
{
   int32_t idx = nir_intrinsic_base(insn) + getIndirect(&insn->src[s], c, indirect);
   if (indirect)
      indirect = mkOp2v(OP_SHL, TYPE_U32, getSSA(4, FILE_ADDRESS), indirect, loadImm(NULL, 4));
   return idx;
}

bool
Converter::run()
{
   bool progress;

   if (prog->dbgFlags & NV50_IR_DEBUG_VERBOSE)
      nir_print_shader(nir, stderr);

   NIR_PASS_V(nir, nir_lower_io, nir_var_all, type_size, (nir_lower_io_options)0);
   NIR_PASS_V(nir, nir_lower_regs_to_ssa);
   NIR_PASS_V(nir, nir_lower_load_const_to_scalar);
   NIR_PASS_V(nir, nir_lower_vars_to_ssa);
   NIR_PASS_V(nir, nir_lower_alu_to_scalar);
   NIR_PASS_V(nir, nir_lower_phis_to_scalar);

   do {
      progress = false;
      NIR_PASS(progress, nir, nir_copy_prop);
      NIR_PASS(progress, nir, nir_opt_remove_phis);
      NIR_PASS(progress, nir, nir_opt_trivial_continues);
      NIR_PASS(progress, nir, nir_opt_cse);
      NIR_PASS(progress, nir, nir_opt_algebraic);
      NIR_PASS(progress, nir, nir_opt_constant_folding);
      NIR_PASS(progress, nir, nir_copy_prop);
      NIR_PASS(progress, nir, nir_opt_dce);
      NIR_PASS(progress, nir, nir_opt_dead_cf);
   } while (progress);

   NIR_PASS_V(nir, nir_lower_locals_to_regs);
   NIR_PASS_V(nir, nir_remove_dead_variables, nir_var_local);
   NIR_PASS_V(nir, nir_convert_from_ssa, true);

   // Garbage collect dead instructions
   nir_sweep(nir);

   if (prog->dbgFlags & NV50_IR_DEBUG_BASIC)
      nir_print_shader(nir, stderr);

   return false;
}

} // unnamed namespace

namespace nv50_ir {

bool
Program::makeFromNIR(struct nv50_ir_prog_info *info)
{
   nir_shader *nir = (nir_shader*)info->bin.source;
   Converter converter(this, nir, info);
   bool result = converter.run();
   if (!result)
      return result;
   LoweringHelper lowering;
   lowering.run(this);
   tlsSize = info->bin.tlsSpace;
   return result;
}

} // namespace nv50_ir
