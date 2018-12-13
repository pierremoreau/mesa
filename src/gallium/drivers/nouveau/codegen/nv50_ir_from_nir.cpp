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

   typedef std::vector<LValue*> LValues;
   typedef unordered_map<unsigned, LValues> NirDefMap;
   typedef unordered_map<unsigned, BasicBlock*> NirBlockMap;

   LValues& convert(nir_alu_dest *);
   BasicBlock* convert(nir_block *);
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

   uint32_t getSlotAddress(nir_intrinsic_instr *, uint8_t idx, uint8_t slot);

   void setInterpolate(nv50_ir_varying *,
                       uint8_t,
                       bool centroid,
                       unsigned semantics);

   Instruction *loadFrom(DataFile, uint8_t, DataType, Value *def, uint32_t base,
                         uint8_t c, Value *indirect0 = NULL,
                         Value *indirect1 = NULL, bool patch = false);
   void storeTo(nir_intrinsic_instr *, DataFile, operation, DataType,
                Value *src, uint8_t idx, uint8_t c, Value *indirect0 = NULL,
                Value *indirect1 = NULL);

   bool isFloatType(nir_alu_type);
   bool isSignedType(nir_alu_type);
   bool isResultFloat(nir_op);
   bool isResultSigned(nir_op);

   DataType getDType(nir_alu_instr *);
   DataType getDType(nir_intrinsic_instr *);
   DataType getDType(nir_op, uint8_t);

   std::vector<DataType> getSTypes(nir_alu_instr *);
   DataType getSType(nir_src &, bool isFloat, bool isSigned);

   bool assignSlots();
   bool parseNIR();

   bool visit(nir_block *);
   bool visit(nir_cf_node *);
   bool visit(nir_function *);
   bool visit(nir_if *);
   bool visit(nir_instr *);
   bool visit(nir_jump_instr *);
   bool visit(nir_loop *);

   nir_shader *nir;

   NirDefMap ssaDefs;
   NirDefMap regDefs;
   NirBlockMap blocks;
   unsigned int curLoopDepth;

   BasicBlock *exit;

   union {
      struct {
         Value *position;
      } fp;
   };
};

Converter::Converter(Program *prog, nir_shader *nir, nv50_ir_prog_info *info)
   : ConverterCommon(prog, info),
     nir(nir),
     curLoopDepth(0) {}

BasicBlock *
Converter::convert(nir_block *block)
{
   NirBlockMap::iterator it = blocks.find(block->index);
   if (it != blocks.end())
      return it->second;

   BasicBlock *bb = new BasicBlock(func);
   blocks[block->index] = bb;
   return bb;
}

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

static void
vert_attrib_to_tgsi_semantic(gl_vert_attrib slot, unsigned *name, unsigned *index)
{
   assert(name && index);

   if (slot >= VERT_ATTRIB_MAX) {
      ERROR("invalid varying slot %u\n", slot);
      assert(false);
      return;
   }

   if (slot >= VERT_ATTRIB_GENERIC0 &&
       slot < VERT_ATTRIB_GENERIC0 + VERT_ATTRIB_GENERIC_MAX) {
      *name = TGSI_SEMANTIC_GENERIC;
      *index = slot - VERT_ATTRIB_GENERIC0;
      return;
   }

   if (slot >= VERT_ATTRIB_TEX0 &&
       slot < VERT_ATTRIB_TEX0 + VERT_ATTRIB_TEX_MAX) {
      *name = TGSI_SEMANTIC_TEXCOORD;
      *index = slot - VERT_ATTRIB_TEX0;
      return;
   }

   switch (slot) {
   case VERT_ATTRIB_COLOR0:
      *name = TGSI_SEMANTIC_COLOR;
      *index = 0;
      break;
   case VERT_ATTRIB_COLOR1:
      *name = TGSI_SEMANTIC_COLOR;
      *index = 1;
      break;
   case VERT_ATTRIB_EDGEFLAG:
      *name = TGSI_SEMANTIC_EDGEFLAG;
      *index = 0;
      break;
   case VERT_ATTRIB_FOG:
      *name = TGSI_SEMANTIC_FOG;
      *index = 0;
      break;
   case VERT_ATTRIB_NORMAL:
      *name = TGSI_SEMANTIC_NORMAL;
      *index = 0;
      break;
   case VERT_ATTRIB_POS:
      *name = TGSI_SEMANTIC_POSITION;
      *index = 0;
      break;
   case VERT_ATTRIB_POINT_SIZE:
      *name = TGSI_SEMANTIC_PSIZE;
      *index = 0;
      break;
   default:
      ERROR("unknown vert attrib slot %u\n", slot);
      assert(false);
      break;
   }
}

static void
varying_slot_to_tgsi_semantic(gl_varying_slot slot, unsigned *name, unsigned *index)
{
   assert(name && index);

   if (slot >= VARYING_SLOT_TESS_MAX) {
      ERROR("invalid varying slot %u\n", slot);
      assert(false);
      return;
   }

   if (slot >= VARYING_SLOT_PATCH0) {
      *name = TGSI_SEMANTIC_PATCH;
      *index = slot - VARYING_SLOT_PATCH0;
      return;
   }

   if (slot >= VARYING_SLOT_VAR0) {
      *name = TGSI_SEMANTIC_GENERIC;
      *index = slot - VARYING_SLOT_VAR0;
      return;
   }

   if (slot >= VARYING_SLOT_TEX0 && slot <= VARYING_SLOT_TEX7) {
      *name = TGSI_SEMANTIC_TEXCOORD;
      *index = slot - VARYING_SLOT_TEX0;
      return;
   }

   switch (slot) {
   case VARYING_SLOT_BFC0:
      *name = TGSI_SEMANTIC_BCOLOR;
      *index = 0;
      break;
   case VARYING_SLOT_BFC1:
      *name = TGSI_SEMANTIC_BCOLOR;
      *index = 1;
      break;
   case VARYING_SLOT_CLIP_DIST0:
      *name = TGSI_SEMANTIC_CLIPDIST;
      *index = 0;
      break;
   case VARYING_SLOT_CLIP_DIST1:
      *name = TGSI_SEMANTIC_CLIPDIST;
      *index = 1;
      break;
   case VARYING_SLOT_CLIP_VERTEX:
      *name = TGSI_SEMANTIC_CLIPVERTEX;
      *index = 0;
      break;
   case VARYING_SLOT_COL0:
      *name = TGSI_SEMANTIC_COLOR;
      *index = 0;
      break;
   case VARYING_SLOT_COL1:
      *name = TGSI_SEMANTIC_COLOR;
      *index = 1;
      break;
   case VARYING_SLOT_EDGE:
      *name = TGSI_SEMANTIC_EDGEFLAG;
      *index = 0;
      break;
   case VARYING_SLOT_FACE:
      *name = TGSI_SEMANTIC_FACE;
      *index = 0;
      break;
   case VARYING_SLOT_FOGC:
      *name = TGSI_SEMANTIC_FOG;
      *index = 0;
      break;
   case VARYING_SLOT_LAYER:
      *name = TGSI_SEMANTIC_LAYER;
      *index = 0;
      break;
   case VARYING_SLOT_PNTC:
      *name = TGSI_SEMANTIC_PCOORD;
      *index = 0;
      break;
   case VARYING_SLOT_POS:
      *name = TGSI_SEMANTIC_POSITION;
      *index = 0;
      break;
   case VARYING_SLOT_PRIMITIVE_ID:
      *name = TGSI_SEMANTIC_PRIMID;
      *index = 0;
      break;
   case VARYING_SLOT_PSIZ:
      *name = TGSI_SEMANTIC_PSIZE;
      *index = 0;
      break;
   case VARYING_SLOT_TESS_LEVEL_INNER:
      *name = TGSI_SEMANTIC_TESSINNER;
      *index = 0;
      break;
   case VARYING_SLOT_TESS_LEVEL_OUTER:
      *name = TGSI_SEMANTIC_TESSOUTER;
      *index = 0;
      break;
   case VARYING_SLOT_VIEWPORT:
      *name = TGSI_SEMANTIC_VIEWPORT_INDEX;
      *index = 0;
      break;
   default:
      ERROR("unknown varying slot %u\n", slot);
      assert(false);
      break;
   }
}

static void
frag_result_to_tgsi_semantic(unsigned slot, unsigned *name, unsigned *index)
{
   if (slot >= FRAG_RESULT_DATA0) {
      *name = TGSI_SEMANTIC_COLOR;
      *index = slot - FRAG_RESULT_COLOR - 2; // intentional
      return;
   }

   switch (slot) {
   case FRAG_RESULT_COLOR:
      *name = TGSI_SEMANTIC_COLOR;
      *index = 0;
      break;
   case FRAG_RESULT_DEPTH:
      *name = TGSI_SEMANTIC_POSITION;
      *index = 0;
      break;
   case FRAG_RESULT_SAMPLE_MASK:
      *name = TGSI_SEMANTIC_SAMPLEMASK;
      *index = 0;
      break;
   default:
      ERROR("unknown frag result slot %u\n", slot);
      assert(false);
      break;
   }
}

// copy of _mesa_sysval_to_semantic
static void
system_val_to_tgsi_semantic(unsigned val, unsigned *name, unsigned *index)
{
   *index = 0;
   switch (val) {
   // Vertex shader
   case SYSTEM_VALUE_VERTEX_ID:
      *name = TGSI_SEMANTIC_VERTEXID;
      break;
   case SYSTEM_VALUE_INSTANCE_ID:
      *name = TGSI_SEMANTIC_INSTANCEID;
      break;
   case SYSTEM_VALUE_VERTEX_ID_ZERO_BASE:
      *name = TGSI_SEMANTIC_VERTEXID_NOBASE;
      break;
   case SYSTEM_VALUE_BASE_VERTEX:
      *name = TGSI_SEMANTIC_BASEVERTEX;
      break;
   case SYSTEM_VALUE_BASE_INSTANCE:
      *name = TGSI_SEMANTIC_BASEINSTANCE;
      break;
   case SYSTEM_VALUE_DRAW_ID:
      *name = TGSI_SEMANTIC_DRAWID;
      break;

   // Geometry shader
   case SYSTEM_VALUE_INVOCATION_ID:
      *name = TGSI_SEMANTIC_INVOCATIONID;
      break;

   // Fragment shader
   case SYSTEM_VALUE_FRAG_COORD:
      *name = TGSI_SEMANTIC_POSITION;
      break;
   case SYSTEM_VALUE_FRONT_FACE:
      *name = TGSI_SEMANTIC_FACE;
      break;
   case SYSTEM_VALUE_SAMPLE_ID:
      *name = TGSI_SEMANTIC_SAMPLEID;
      break;
   case SYSTEM_VALUE_SAMPLE_POS:
      *name = TGSI_SEMANTIC_SAMPLEPOS;
      break;
   case SYSTEM_VALUE_SAMPLE_MASK_IN:
      *name = TGSI_SEMANTIC_SAMPLEMASK;
      break;
   case SYSTEM_VALUE_HELPER_INVOCATION:
      *name = TGSI_SEMANTIC_HELPER_INVOCATION;
      break;

   // Tessellation shader
   case SYSTEM_VALUE_TESS_COORD:
      *name = TGSI_SEMANTIC_TESSCOORD;
      break;
   case SYSTEM_VALUE_VERTICES_IN:
      *name = TGSI_SEMANTIC_VERTICESIN;
      break;
   case SYSTEM_VALUE_PRIMITIVE_ID:
      *name = TGSI_SEMANTIC_PRIMID;
      break;
   case SYSTEM_VALUE_TESS_LEVEL_OUTER:
      *name = TGSI_SEMANTIC_TESSOUTER;
      break;
   case SYSTEM_VALUE_TESS_LEVEL_INNER:
      *name = TGSI_SEMANTIC_TESSINNER;
      break;

   // Compute shader
   case SYSTEM_VALUE_LOCAL_INVOCATION_ID:
      *name = TGSI_SEMANTIC_THREAD_ID;
      break;
   case SYSTEM_VALUE_WORK_GROUP_ID:
      *name = TGSI_SEMANTIC_BLOCK_ID;
      break;
   case SYSTEM_VALUE_NUM_WORK_GROUPS:
      *name = TGSI_SEMANTIC_GRID_SIZE;
      break;
   case SYSTEM_VALUE_LOCAL_GROUP_SIZE:
      *name = TGSI_SEMANTIC_BLOCK_SIZE;
      break;

   // ARB_shader_ballot
   case SYSTEM_VALUE_SUBGROUP_SIZE:
      *name = TGSI_SEMANTIC_SUBGROUP_SIZE;
      break;
   case SYSTEM_VALUE_SUBGROUP_INVOCATION:
      *name = TGSI_SEMANTIC_SUBGROUP_INVOCATION;
      break;
   case SYSTEM_VALUE_SUBGROUP_EQ_MASK:
      *name = TGSI_SEMANTIC_SUBGROUP_EQ_MASK;
      break;
   case SYSTEM_VALUE_SUBGROUP_GE_MASK:
      *name = TGSI_SEMANTIC_SUBGROUP_GE_MASK;
      break;
   case SYSTEM_VALUE_SUBGROUP_GT_MASK:
      *name = TGSI_SEMANTIC_SUBGROUP_GT_MASK;
      break;
   case SYSTEM_VALUE_SUBGROUP_LE_MASK:
      *name = TGSI_SEMANTIC_SUBGROUP_LE_MASK;
      break;
   case SYSTEM_VALUE_SUBGROUP_LT_MASK:
      *name = TGSI_SEMANTIC_SUBGROUP_LT_MASK;
      break;

   default:
      ERROR("unknown system value %u\n", val);
      assert(false);
      break;
   }
}

void
Converter::setInterpolate(nv50_ir_varying *var,
                          uint8_t mode,
                          bool centroid,
                          unsigned semantic)
{
   switch (mode) {
   case INTERP_MODE_FLAT:
      var->flat = 1;
      break;
   case INTERP_MODE_NONE:
      if (semantic == TGSI_SEMANTIC_COLOR)
         var->sc = 1;
      else if (semantic == TGSI_SEMANTIC_POSITION)
         var->linear = 1;
      break;
   case INTERP_MODE_NOPERSPECTIVE:
      var->linear = 1;
      break;
   case INTERP_MODE_SMOOTH:
      break;
   }
   var->centroid = centroid;
}

static uint16_t
calcSlots(const glsl_type *type, Program::Type stage, const shader_info &info,
          bool input, const nir_variable *var)
{
   if (!type->is_array())
      return type->count_attribute_slots(false);

   uint16_t slots;
   switch (stage) {
   case Program::TYPE_GEOMETRY:
      slots = type->uniform_locations();
      if (input)
         slots /= info.gs.vertices_in;
      break;
   case Program::TYPE_TESSELLATION_CONTROL:
   case Program::TYPE_TESSELLATION_EVAL:
      // remove first dimension
      if (var->data.patch || (!input && stage == Program::TYPE_TESSELLATION_EVAL))
         slots = type->uniform_locations();
      else
         slots = type->fields.array->uniform_locations();
      break;
   default:
      slots = type->count_attribute_slots(false);
      break;
   }

   return slots;
}

bool Converter::assignSlots() {
   unsigned name;
   unsigned index;

   info->io.viewportId = -1;
   info->numInputs = 0;

   // we have to fixup the uniform locations for arrays
   unsigned numImages = 0;
   nir_foreach_variable(var, &nir->uniforms) {
      const glsl_type *type = var->type;
      if (!type->without_array()->is_image())
         continue;
      var->data.driver_location = numImages;
      numImages += type->is_array() ? type->arrays_of_arrays_size() : 1;
   }

   nir_foreach_variable(var, &nir->inputs) {
      const glsl_type *type = var->type;
      int slot = var->data.location;
      uint16_t slots = calcSlots(type, prog->getType(), nir->info, true, var);
      uint32_t comp = type->is_array() ? type->without_array()->component_slots()
                                       : type->component_slots();
      uint32_t frac = var->data.location_frac;
      uint32_t vary = var->data.driver_location;

      if (glsl_base_type_is_64bit(type->without_array()->base_type)) {
         if (comp > 2)
            slots *= 2;
      }

      assert(vary + slots <= PIPE_MAX_SHADER_INPUTS);

      switch(prog->getType()) {
      case Program::TYPE_FRAGMENT:
         varying_slot_to_tgsi_semantic((gl_varying_slot)slot, &name, &index);
         for (uint16_t i = 0; i < slots; ++i) {
            setInterpolate(&info->in[vary + i], var->data.interpolation,
                           var->data.centroid | var->data.sample, name);
         }
         break;
      case Program::TYPE_GEOMETRY:
         varying_slot_to_tgsi_semantic((gl_varying_slot)slot, &name, &index);
         break;
      case Program::TYPE_TESSELLATION_CONTROL:
      case Program::TYPE_TESSELLATION_EVAL:
         varying_slot_to_tgsi_semantic((gl_varying_slot)slot, &name, &index);
         if (var->data.patch && name == TGSI_SEMANTIC_PATCH)
            info->numPatchConstants = MAX2(info->numPatchConstants, index + slots);
         break;
      case Program::TYPE_VERTEX:
         vert_attrib_to_tgsi_semantic((gl_vert_attrib)slot, &name, &index);
         switch (name) {
         case TGSI_SEMANTIC_EDGEFLAG:
            info->io.edgeFlagIn = vary;
            break;
         default:
            break;
         }
         break;
      default:
         ERROR("unknown shader type %u in assignSlots\n", prog->getType());
         return false;
      }

      for (uint16_t i = 0u; i < slots; ++i, ++vary) {
         info->in[vary].id = vary;
         info->in[vary].patch = var->data.patch;
         info->in[vary].sn = name;
         info->in[vary].si = index + i;
         if (glsl_base_type_is_64bit(type->without_array()->base_type))
            if (i & 0x1)
               info->in[vary].mask |= (((1 << (comp * 2)) - 1) << (frac * 2) >> 0x4);
            else
               info->in[vary].mask |= (((1 << (comp * 2)) - 1) << (frac * 2) & 0xf);
         else
            info->in[vary].mask |= ((1 << comp) - 1) << frac;
      }
      info->numInputs = std::max<uint8_t>(info->numInputs, vary);
   }

   info->numOutputs = 0;
   nir_foreach_variable(var, &nir->outputs) {
      const glsl_type *type = var->type;
      int slot = var->data.location;
      uint16_t slots = calcSlots(type, prog->getType(), nir->info, false, var);
      uint32_t comp = type->is_array() ? type->without_array()->component_slots()
                                       : type->component_slots();
      uint32_t frac = var->data.location_frac;
      uint32_t vary = var->data.driver_location;

      if (glsl_base_type_is_64bit(type->without_array()->base_type)) {
         if (comp > 2)
            slots *= 2;
      }

      assert(vary < PIPE_MAX_SHADER_OUTPUTS);

      switch(prog->getType()) {
      case Program::TYPE_FRAGMENT:
         frag_result_to_tgsi_semantic((gl_frag_result)slot, &name, &index);
         switch (name) {
         case TGSI_SEMANTIC_COLOR:
            if (!var->data.fb_fetch_output)
               info->prop.fp.numColourResults++;
            info->prop.fp.separateFragData = true;
            // sometimes we get FRAG_RESULT_DATAX with data.index 0
            // sometimes we get FRAG_RESULT_DATA0 with data.index X
            index = index == 0 ? var->data.index : index;
            break;
         case TGSI_SEMANTIC_POSITION:
            info->io.fragDepth = vary;
            info->prop.fp.writesDepth = true;
            break;
         case TGSI_SEMANTIC_SAMPLEMASK:
            info->io.sampleMask = vary;
            break;
         default:
            break;
         }
         break;
      case Program::TYPE_GEOMETRY:
      case Program::TYPE_TESSELLATION_CONTROL:
      case Program::TYPE_TESSELLATION_EVAL:
      case Program::TYPE_VERTEX:
         varying_slot_to_tgsi_semantic((gl_varying_slot)slot, &name, &index);

         if (var->data.patch && name != TGSI_SEMANTIC_TESSINNER &&
             name != TGSI_SEMANTIC_TESSOUTER)
            info->numPatchConstants = MAX2(info->numPatchConstants, index + slots);

         switch (name) {
         case TGSI_SEMANTIC_CLIPDIST:
            info->io.genUserClip = -1;
            break;
         case TGSI_SEMANTIC_EDGEFLAG:
            info->io.edgeFlagOut = vary;
            break;
         default:
            break;
         }
         break;
      default:
         ERROR("unknown shader type %u in assignSlots\n", prog->getType());
         return false;
      }

      for (uint16_t i = 0u; i < slots; ++i, ++vary) {
         info->out[vary].id = vary;
         info->out[vary].patch = var->data.patch;
         info->out[vary].sn = name;
         info->out[vary].si = index + i;
         if (glsl_base_type_is_64bit(type->without_array()->base_type))
            if (i & 0x1)
               info->out[vary].mask |= (((1 << (comp * 2)) - 1) << (frac * 2) >> 0x4);
            else
               info->out[vary].mask |= (((1 << (comp * 2)) - 1) << (frac * 2) & 0xf);
         else
            info->out[vary].mask |= ((1 << comp) - 1) << frac;

         if (nir->info.outputs_read & 1ll << slot)
            info->out[vary].oread = 1;
      }
      info->numOutputs = std::max<uint8_t>(info->numOutputs, vary);
   }

   info->numSysVals = 0;
   for (uint8_t i = 0; i < 64; ++i) {
      if (!(nir->info.system_values_read & 1ll << i))
         continue;

      system_val_to_tgsi_semantic(i, &name, &index);
      info->sv[info->numSysVals].sn = name;
      info->sv[info->numSysVals].si = index;
      info->sv[info->numSysVals].input = 0; // TODO inferSysValDirection(sn);

      switch (i) {
      case SYSTEM_VALUE_INSTANCE_ID:
         info->io.instanceId = info->numSysVals;
         break;
      case SYSTEM_VALUE_TESS_LEVEL_INNER:
      case SYSTEM_VALUE_TESS_LEVEL_OUTER:
         info->sv[info->numSysVals].patch = 1;
         break;
      case SYSTEM_VALUE_VERTEX_ID:
         info->io.vertexId = info->numSysVals;
         break;
      default:
         break;
      }

      info->numSysVals += 1;
   }

   if (info->io.genUserClip > 0) {
      info->io.clipDistances = info->io.genUserClip;

      const unsigned int nOut = (info->io.genUserClip + 3) / 4;

      for (unsigned int n = 0; n < nOut; ++n) {
         unsigned int i = info->numOutputs++;
         info->out[i].id = i;
         info->out[i].sn = TGSI_SEMANTIC_CLIPDIST;
         info->out[i].si = n;
         info->out[i].mask = ((1 << info->io.clipDistances) - 1) >> (n * 4);
      }
   }

   return info->assignSlots(info) == 0;
}

uint32_t
Converter::getSlotAddress(nir_intrinsic_instr *insn, uint8_t idx, uint8_t slot)
{
   DataType ty;
   int offset = nir_intrinsic_component(insn);
   bool input;

   if (nir_intrinsic_infos[insn->intrinsic].has_dest)
      ty = getDType(insn);
   else
      ty = getSType(insn->src[0], false, false);

   switch (insn->intrinsic) {
   case nir_intrinsic_load_input:
   case nir_intrinsic_load_interpolated_input:
   case nir_intrinsic_load_per_vertex_input:
      input = true;
      break;
   case nir_intrinsic_load_output:
   case nir_intrinsic_load_per_vertex_output:
   case nir_intrinsic_store_output:
   case nir_intrinsic_store_per_vertex_output:
      input = false;
      break;
   default:
      ERROR("unknown intrinsic in getSlotAddress %s",
            nir_intrinsic_infos[insn->intrinsic].name);
      input = false;
      assert(false);
      break;
   }

   if (typeSizeof(ty) == 8) {
      slot *= 2;
      slot += offset;
      if (slot >= 4) {
         idx += 1;
         slot -= 4;
      }
   } else {
      slot += offset;
   }

   assert(slot < 4);
   assert(!input || idx < PIPE_MAX_SHADER_INPUTS);
   assert(input || idx < PIPE_MAX_SHADER_OUTPUTS);

   const nv50_ir_varying *vary = input ? info->in : info->out;
   return vary[idx].slot[slot] * 4;
}

Instruction *
Converter::loadFrom(DataFile file, uint8_t i, DataType ty, Value *def,
                    uint32_t base, uint8_t c, Value *indirect0,
                    Value *indirect1, bool patch)
{
   unsigned int tySize = typeSizeof(ty);

   if (tySize == 8 &&
       (file == FILE_MEMORY_CONST || file == FILE_MEMORY_BUFFER || indirect0)) {
      Value *lo = getSSA();
      Value *hi = getSSA();

      Instruction *loi =
         mkLoad(TYPE_U32, lo,
                mkSymbol(file, i, TYPE_U32, base + c * tySize),
                indirect0);
      loi->setIndirect(0, 1, indirect1);
      loi->perPatch = patch;

      Instruction *hii =
         mkLoad(TYPE_U32, hi,
                mkSymbol(file, i, TYPE_U32, base + c * tySize + 4),
                indirect0);
      hii->setIndirect(0, 1, indirect1);
      hii->perPatch = patch;

      return mkOp2(OP_MERGE, ty, def, lo, hi);
   } else {
      Instruction *ld =
         mkLoad(ty, def, mkSymbol(file, i, ty, base + c * tySize), indirect0);
      ld->setIndirect(0, 1, indirect1);
      ld->perPatch = patch;
      return ld;
   }
}

void
Converter::storeTo(nir_intrinsic_instr *insn, DataFile file, operation op,
                   DataType ty, Value *src, uint8_t idx, uint8_t c,
                   Value *indirect0, Value *indirect1)
{
   uint8_t size = typeSizeof(ty);
   uint32_t address = getSlotAddress(insn, idx, c);

   if (size == 8 && indirect0) {
      Value *split[2];
      mkSplit(split, 4, src);

      if (op == OP_EXPORT) {
         split[0] = mkMov(getSSA(), split[0], ty)->getDef(0);
         split[1] = mkMov(getSSA(), split[1], ty)->getDef(0);
      }

      mkStore(op, TYPE_U32, mkSymbol(file, 0, TYPE_U32, address), indirect0,
              split[0])->perPatch = info->out[idx].patch;
      mkStore(op, TYPE_U32, mkSymbol(file, 0, TYPE_U32, address + 4), indirect0,
              split[1])->perPatch = info->out[idx].patch;
   } else {
      if (op == OP_EXPORT)
         src = mkMov(getSSA(size), src, ty)->getDef(0);
      mkStore(op, ty, mkSymbol(file, 0, ty, address), indirect0,
              src)->perPatch = info->out[idx].patch;
   }
}

bool
Converter::parseNIR()
{
   info->io.clipDistances = nir->info.clip_distance_array_size;
   info->io.cullDistances = nir->info.cull_distance_array_size;

   switch(prog->getType()) {
   case Program::TYPE_COMPUTE:
      info->prop.cp.numThreads[0] = nir->info.cs.local_size[0];
      info->prop.cp.numThreads[1] = nir->info.cs.local_size[1];
      info->prop.cp.numThreads[2] = nir->info.cs.local_size[2];
      info->bin.smemSize = nir->info.cs.shared_size;
      break;
   case Program::TYPE_FRAGMENT:
      info->prop.fp.earlyFragTests = nir->info.fs.early_fragment_tests;
      info->prop.fp.persampleInvocation =
         (nir->info.system_values_read & SYSTEM_BIT_SAMPLE_ID) ||
         (nir->info.system_values_read & SYSTEM_BIT_SAMPLE_POS);
      info->prop.fp.postDepthCoverage = nir->info.fs.post_depth_coverage;
      info->prop.fp.readsSampleLocations =
         (nir->info.system_values_read & SYSTEM_BIT_SAMPLE_POS);
      info->prop.fp.usesDiscard = nir->info.fs.uses_discard;
      info->prop.fp.usesSampleMaskIn =
         !!(nir->info.system_values_read & SYSTEM_BIT_SAMPLE_MASK_IN);
      break;
   case Program::TYPE_GEOMETRY:
      info->prop.gp.inputPrim = nir->info.gs.input_primitive;
      info->prop.gp.instanceCount = nir->info.gs.invocations;
      info->prop.gp.maxVertices = nir->info.gs.vertices_out;
      info->prop.gp.outputPrim = nir->info.gs.output_primitive;
      break;
   case Program::TYPE_TESSELLATION_CONTROL:
   case Program::TYPE_TESSELLATION_EVAL:
      if (nir->info.tess.primitive_mode == GL_ISOLINES)
         info->prop.tp.domain = GL_LINES;
      else
         info->prop.tp.domain = nir->info.tess.primitive_mode;
      info->prop.tp.outputPatchSize = nir->info.tess.tcs_vertices_out;
      info->prop.tp.outputPrim =
         nir->info.tess.point_mode ? PIPE_PRIM_POINTS : PIPE_PRIM_TRIANGLES;
      info->prop.tp.partitioning = (nir->info.tess.spacing + 1) % 3;
      info->prop.tp.winding = !nir->info.tess.ccw;
      break;
   case Program::TYPE_VERTEX:
      info->prop.vp.usesDrawParameters =
         (nir->info.system_values_read & BITFIELD64_BIT(SYSTEM_VALUE_BASE_VERTEX)) ||
         (nir->info.system_values_read & BITFIELD64_BIT(SYSTEM_VALUE_BASE_INSTANCE)) ||
         (nir->info.system_values_read & BITFIELD64_BIT(SYSTEM_VALUE_DRAW_ID));
      break;
   default:
      break;
   }

   return true;
}

bool
Converter::visit(nir_function *function)
{
   // we only support emiting the main function for now
   assert(!strcmp(function->name, "main"));
   assert(function->impl);

   // usually the blocks will set everything up, but main is special
   BasicBlock *entry = new BasicBlock(prog->main);
   exit = new BasicBlock(prog->main);
   blocks[nir_start_block(function->impl)->index] = entry;
   prog->main->setEntry(entry);
   prog->main->setExit(exit);

   setPosition(entry, true);

   switch (prog->getType()) {
   case Program::TYPE_TESSELLATION_CONTROL:
      outBase = mkOp2v(
         OP_SUB, TYPE_U32, getSSA(),
         mkOp1v(OP_RDSV, TYPE_U32, getSSA(), mkSysVal(SV_LANEID, 0)),
         mkOp1v(OP_RDSV, TYPE_U32, getSSA(), mkSysVal(SV_INVOCATION_ID, 0)));
      break;
   case Program::TYPE_FRAGMENT: {
      Symbol *sv = mkSysVal(SV_POSITION, 3);
      fragCoord[3] = mkOp1v(OP_RDSV, TYPE_F32, getSSA(), sv);
      fp.position = mkOp1v(OP_RCP, TYPE_F32, fragCoord[3], fragCoord[3]);
      break;
   }
   default:
      break;
   }

   nir_index_ssa_defs(function->impl);
   foreach_list_typed(nir_cf_node, node, node, &function->impl->body) {
      if (!visit(node))
         return false;
   }

   bb->cfg.attach(&exit->cfg, Graph::Edge::TREE);
   setPosition(exit, true);

   // TODO: for non main function this needs to be a OP_RETURN
   mkOp(OP_EXIT, TYPE_NONE, NULL)->terminator = 1;
   return true;
}

bool
Converter::visit(nir_cf_node *node)
{
   switch (node->type) {
   case nir_cf_node_block:
      return visit(nir_cf_node_as_block(node));
   case nir_cf_node_if:
      return visit(nir_cf_node_as_if(node));
   case nir_cf_node_loop:
      return visit(nir_cf_node_as_loop(node));
   default:
      ERROR("unknown nir_cf_node type %u\n", node->type);
      return false;
   }
}

bool
Converter::visit(nir_block *block)
{
   if (!block->predecessors->entries && block->instr_list.is_empty())
      return true;

   BasicBlock *bb = convert(block);

   setPosition(bb, true);
   nir_foreach_instr(insn, block) {
      if (!visit(insn))
         return false;
   }
   return true;
}

bool
Converter::visit(nir_if *nif)
{
   DataType sType = getSType(nif->condition, false, false);
   Value *src = getSrc(&nif->condition, 0);

   nir_block *lastThen = nir_if_last_then_block(nif);
   nir_block *lastElse = nir_if_last_else_block(nif);

   assert(!lastThen->successors[1]);
   assert(!lastElse->successors[1]);

   BasicBlock *ifBB = convert(nir_if_first_then_block(nif));
   BasicBlock *elseBB = convert(nir_if_first_else_block(nif));

   bb->cfg.attach(&ifBB->cfg, Graph::Edge::TREE);
   bb->cfg.attach(&elseBB->cfg, Graph::Edge::TREE);

   // we only insert joinats, if both nodes end up at the end of the if again.
   // the reason for this to not happens are breaks/continues/ret/... which
   // have their own handling
   if (lastThen->successors[0] == lastElse->successors[0])
      bb->joinAt = mkFlow(OP_JOINAT, convert(lastThen->successors[0]),
                          CC_ALWAYS, NULL);

   mkFlow(OP_BRA, elseBB, CC_EQ, src)->setType(sType);

   foreach_list_typed(nir_cf_node, node, node, &nif->then_list) {
      if (!visit(node))
         return false;
   }
   setPosition(convert(lastThen), true);
   if (!bb->getExit() ||
       !bb->getExit()->asFlow() ||
        bb->getExit()->asFlow()->op == OP_JOIN) {
      BasicBlock *tailBB = convert(lastThen->successors[0]);
      mkFlow(OP_BRA, tailBB, CC_ALWAYS, NULL);
      bb->cfg.attach(&tailBB->cfg, Graph::Edge::FORWARD);
   }

   foreach_list_typed(nir_cf_node, node, node, &nif->else_list) {
      if (!visit(node))
         return false;
   }
   setPosition(convert(lastElse), true);
   if (!bb->getExit() ||
       !bb->getExit()->asFlow() ||
        bb->getExit()->asFlow()->op == OP_JOIN) {
      BasicBlock *tailBB = convert(lastElse->successors[0]);
      mkFlow(OP_BRA, tailBB, CC_ALWAYS, NULL);
      bb->cfg.attach(&tailBB->cfg, Graph::Edge::FORWARD);
   }

   if (lastThen->successors[0] == lastElse->successors[0]) {
      setPosition(convert(lastThen->successors[0]), true);
      mkFlow(OP_JOIN, NULL, CC_ALWAYS, NULL)->fixed = 1;
   }

   return true;
}

bool
Converter::visit(nir_loop *loop)
{
   curLoopDepth += 1;
   func->loopNestingBound = std::max(func->loopNestingBound, curLoopDepth);

   BasicBlock *loopBB = convert(nir_loop_first_block(loop));
   BasicBlock *tailBB =
      convert(nir_cf_node_as_block(nir_cf_node_next(&loop->cf_node)));
   bb->cfg.attach(&loopBB->cfg, Graph::Edge::TREE);

   mkFlow(OP_PREBREAK, tailBB, CC_ALWAYS, NULL);
   setPosition(loopBB, false);
   mkFlow(OP_PRECONT, loopBB, CC_ALWAYS, NULL);

   foreach_list_typed(nir_cf_node, node, node, &loop->body) {
      if (!visit(node))
         return false;
   }
   Instruction *insn = bb->getExit();
   if (bb->cfg.incidentCount() != 0) {
      if (!insn || !insn->asFlow()) {
         mkFlow(OP_CONT, loopBB, CC_ALWAYS, NULL);
         bb->cfg.attach(&loopBB->cfg, Graph::Edge::BACK);
      } else if (insn && insn->op == OP_BRA && !insn->getPredicate() &&
                 tailBB->cfg.incidentCount() == 0) {
         // RA doesn't like having blocks around with no incident edge,
         // so we create a fake one to make it happy
         bb->cfg.attach(&tailBB->cfg, Graph::Edge::TREE);
      }
   }

   curLoopDepth -= 1;

   return true;
}

bool
Converter::visit(nir_instr *insn)
{
   switch (insn->type) {
   case nir_instr_type_jump:
      return visit(nir_instr_as_jump(insn));
   default:
      ERROR("unknown nir_instr type %u\n", insn->type);
      return false;
   }
   return true;
}

bool
Converter::visit(nir_jump_instr *insn)
{
   switch (insn->type) {
   case nir_jump_return:
      // TODO: this only works in the main function
      mkFlow(OP_BRA, exit, CC_ALWAYS, NULL);
      bb->cfg.attach(&exit->cfg, Graph::Edge::CROSS);
      break;
   case nir_jump_break:
   case nir_jump_continue: {
      bool isBreak = insn->type == nir_jump_break;
      nir_block *block = insn->instr.block;
      assert(!block->successors[1]);
      BasicBlock *target = convert(block->successors[0]);
      mkFlow(isBreak ? OP_BREAK : OP_CONT, target, CC_ALWAYS, NULL);
      bb->cfg.attach(&target->cfg, isBreak ? Graph::Edge::CROSS : Graph::Edge::BACK);
      break;
   }
   default:
      ERROR("unknown nir_jump_type %u\n", insn->type);
      return false;
   }

   return true;
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

   if (!parseNIR()) {
      ERROR("Couldn't prase NIR!\n");
      return false;
   }

   if (!assignSlots()) {
      ERROR("Couldn't assign slots!\n");
      return false;
   }

   if (prog->dbgFlags & NV50_IR_DEBUG_BASIC)
      nir_print_shader(nir, stderr);

   nir_foreach_function(function, nir) {
      if (!visit(function))
         return false;
   }

   return true;
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
