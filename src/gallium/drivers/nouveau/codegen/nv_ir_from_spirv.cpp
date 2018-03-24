/*
 * Copyright 2017 Pierre Moreau
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
 */

#include <cstring>
#include <functional>
#include <set>
#include <stack>
#include <unordered_map>
#include <vector>

#include "codegen/nv50_ir.h"
#include "codegen/nv50_ir_util.h"
#include "codegen/nv50_ir_build_util.h"

#include "spirv.hpp11"

namespace spirv {

using word = unsigned int;
using Words = std::vector<word>;
using Ids = std::vector<spv::Id>;

static inline bool hasFlag(spv::ImageOperandsMask v, spv::ImageOperandsShift f) { return static_cast<uint32_t>(v) & (1u << static_cast<uint32_t>(f)); }
static inline bool hasFlag(spv::FPFastMathModeMask v, spv::FPFastMathModeShift f) { return static_cast<uint32_t>(v) & (1u << static_cast<uint32_t>(f)); }
static inline bool hasFlag(spv::SelectionControlMask v, spv::SelectionControlShift f) { return static_cast<uint32_t>(v) & (1u << static_cast<uint32_t>(f)); }
static inline bool hasFlag(spv::LoopControlMask v, spv::LoopControlShift f) { return static_cast<uint32_t>(v) & (1u << static_cast<uint32_t>(f)); }
static inline bool hasFlag(spv::FunctionControlMask v, spv::FunctionControlShift f) { return static_cast<uint32_t>(v) & (1u << static_cast<uint32_t>(f)); }
static inline bool hasFlag(spv::MemorySemanticsMask v, spv::MemorySemanticsShift f) { return static_cast<uint32_t>(v) & (1u << static_cast<uint32_t>(f)); }
static inline bool hasFlag(spv::MemoryAccessMask v, spv::MemoryAccessShift f) { return static_cast<uint32_t>(v) & (1u << static_cast<uint32_t>(f)); }
static inline bool hasFlag(spv::KernelProfilingInfoMask v, spv::KernelProfilingInfoShift f) { return static_cast<uint32_t>(v) & (1u << static_cast<uint32_t>(f)); }

template<typename T>
T getWord(const char *binary, unsigned word_offset)
{
   return static_cast<T>(reinterpret_cast<const uint32_t*>(binary)[word_offset]);
}

} // namespace spirv


namespace {

using namespace spirv;
using namespace nv50_ir;

class Converter : public BuildUtil
{
public:
   struct EntryPoint {
      uint32_t index;
      spv::ExecutionModel executionModel;
      std::string name;
      std::vector<spv::Id> interface;
   };
   enum class SpirvFile { NONE, TEMPORARY, SHARED, GLOBAL, CONST, INPUT, PREDICATE, IMMEDIATE };
   using Decoration = std::unordered_map<spv::Decoration, std::vector<Words>>;
   using Decorations = std::unordered_map<spv::Id, Decoration>;
    struct PValue {
      union { Value *value; Value *indirect; };
      PValue() : value(nullptr), symbol(nullptr) {}
      Symbol *symbol;
      PValue(Value *value) : value(value), symbol(nullptr) {}
      PValue(Symbol *symbol, Value *indirect) : symbol(symbol), indirect(indirect) {}
      bool isUndefined() const { return symbol == nullptr && value == nullptr; }
      bool isValue() const { return value != nullptr && value->reg.file == FILE_GPR; }
   };
   class Type {
   public:
      Type(spv::Op type) : type(type), id(0u) {}
      virtual ~Type() {}
      spv::Id getId() const { return id; }
      virtual spv::Op getType() const { return type; }
      virtual bool isBasicType() const { return false; }
      virtual bool isCompooundType() const { return false; }
      virtual bool isVoidType() const { return false; }
      virtual std::vector<ImmediateValue *> generateConstant(const Words &words,
                                                             Converter &conv, unsigned int& position) const { assert(false); return std::vector<ImmediateValue *>(); }
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const = 0;
      virtual unsigned int getSize(void) const { assert(false); return 0u; }
      unsigned int getAlignment() const { return alignment; }
      virtual std::vector<unsigned int> getPaddings() const { return { 0u }; }
      virtual enum DataType getEnumType(int isSigned = -1) const { assert(false); return DataType::TYPE_NONE; }
      virtual unsigned int getElementsNb(void) const { return 1u; }
      virtual unsigned int getElementSize(unsigned int /*index*/) const { return getSize(); }
      virtual Type const* getElementType(unsigned int /*index*/) const { return this; }
      virtual enum DataType getElementEnumType(unsigned int /*index*/, int isSigned = -1) const { return getEnumType(isSigned); }
      virtual unsigned int getGlobalIdx(std::vector<unsigned int> const& elementIds, unsigned position = 0u) const { return 0u; }
      virtual void getGlobalOffset(BuildUtil *bu, Decoration const& decoration, Value *offset, std::vector<Value *> ids, unsigned position = 0u) const { if (position < ids.size()) assert(false); }
      virtual bool isVectorOfSize(unsigned int /*size*/) const { return false; }
      virtual bool isUInt() const { return false; }

      const spv::Op type;
      spv::Id id;
      unsigned int alignment;
   };
   struct SpirVValue {
      SpirvFile storageFile;
      Type const* type;
      std::vector<PValue> value;
      std::vector<unsigned int> paddings; // How to align each component: this will be used by OpCopyMemory* for example
      bool is_packed;
   };
   using ValueMap = std::unordered_map<spv::Id, SpirVValue>;
   class TypeVoid : public Type {
   public:
      TypeVoid(unsigned int numWords, unsigned int firstWord,
               const char *const binary, bool &didSucceed);
      virtual ~TypeVoid() {}
      virtual bool isVoidType() const override { return true; }
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override { assert(false); return std::vector<Value *>(); }
   };
   class TypeBool : public Type {
   public:
      TypeBool(unsigned int numWords, unsigned int firstWord,
              const char *const binary, bool &didSucceed);
      virtual ~TypeBool() {}
      virtual bool isBasicType() const override { return true; }
      virtual std::vector<ImmediateValue *> generateConstant(const Words &words,
                                                             Converter &conv, unsigned int& position) const override;
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override;
      virtual unsigned int getSize(void) const override { return 1u; } // XXX no idea
      virtual enum DataType getEnumType(int isSigned = -1) const override;
   };
   class TypeInt : public Type {
   public:
      TypeInt(unsigned int numWords, unsigned int firstWord,
              const char *const binary, bool &didSucceed);
      virtual ~TypeInt() {}
      virtual bool isBasicType() const override { return true; }
      virtual std::vector<ImmediateValue *> generateConstant(const Words &words,
                                                             Converter &conv, unsigned int& position) const override;
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override;
      virtual unsigned int getSize(void) const override { return static_cast<uint32_t>(width) / 8u; }
      virtual enum DataType getEnumType(int isSigned = -1) const override;
      virtual bool isUInt() const override { return !static_cast<bool>(signedness); }

      word width;
      word signedness;
   };
   class TypeFloat : public Type {
   public:
      TypeFloat(unsigned int numWords, unsigned int firstWord,
                const char *const binary, bool &didSucceed);
      virtual ~TypeFloat() {}
      virtual bool isBasicType() const override { return true; }
      virtual std::vector<ImmediateValue *> generateConstant(const Words &words,
                                                             Converter &conv, unsigned int& position) const override;
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override;
      virtual unsigned int getSize(void) const override { return static_cast<uint32_t>(width) / 8u; }
      virtual enum DataType getEnumType(int isSigned = -1) const override;

      word width;
   };
   class TypeStruct : public Type {
   public:
      TypeStruct(unsigned int numWords, unsigned int firstWord,
                 const char *const binary, bool &didSucceed, std::unordered_map<spv::Id, Type*> const& types,
                 Decorations const& decorations);
      virtual ~TypeStruct() {}
      virtual bool isCompooundType() const override { return true; }
      virtual std::vector<ImmediateValue *> generateConstant(const Words &words,
                                                             Converter &conv, unsigned int& position) const override;
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override;
      virtual unsigned int getSize(void) const override { return size; }
      virtual enum DataType getEnumType(int isSigned = -1) const override;
      virtual unsigned int getElementsNb(void) const override { return static_cast<unsigned>(member_ids.size()); }
      virtual unsigned int getElementSize(unsigned int index) const override;
      virtual Type const* getElementType(unsigned int index) const override;
      virtual enum DataType getElementEnumType(unsigned int index, int isSigned = -1) const override;
      virtual unsigned int getGlobalIdx(std::vector<unsigned int> const& elementIds, unsigned position = 0u) const override;
      virtual void getGlobalOffset(BuildUtil *bu, Decoration const& decoration, Value *offset, std::vector<Value *> ids, unsigned position = 0u) const override;
      virtual std::vector<unsigned int> getPaddings() const override { return member_paddings; }

      std::vector<spv::Id> member_ids;
      std::vector<Type*> member_types;
      std::vector<unsigned int> member_paddings;
      unsigned size;
   };
   class TypeVector : public Type {
   public:
      TypeVector(unsigned int numWords, unsigned int firstWord,
                 const char *const binary, bool &didSucceed, std::unordered_map<spv::Id, Type*> const& types);
      virtual ~TypeVector() {}
      virtual bool isCompooundType() const override { return true; }
      virtual std::vector<ImmediateValue *> generateConstant(const Words &words,
                                                             Converter &conv, unsigned int& position) const override;
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override;
      virtual unsigned int getSize(void) const override;
      virtual enum DataType getEnumType(int isSigned = -1) const override;
      virtual unsigned int getElementsNb(void) const override { return static_cast<unsigned>(elements_nb); }
      virtual unsigned int getElementSize(unsigned int /*index*/) const override;
      virtual Type const* getElementType(unsigned int /*index*/) const override;
      virtual enum DataType getElementEnumType(unsigned int /*index*/, int isSigned = -1) const override;
      virtual unsigned int getGlobalIdx(std::vector<unsigned int> const& elementIds, unsigned position = 0u) const override;
      virtual void getGlobalOffset(BuildUtil *bu, Decoration const& decoration, Value *offset, std::vector<Value *> ids, unsigned position = 0u) const override;
      virtual std::vector<unsigned int> getPaddings() const override;
      virtual bool isVectorOfSize(unsigned int size) const override { return size == elements_nb; }

      spv::Id component_type_id;
      Type* component_type;
      word elements_nb;
   };
   class TypeArray : public Type {
   public:
      TypeArray(unsigned int numWords, unsigned int firstWord,
                 const char *const binary, bool &didSucceed, std::unordered_map<spv::Id, Type*> const& types,
                 const ValueMap &m);
      virtual ~TypeArray() {}
      virtual bool isCompooundType() const override { return true; }
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override;
      virtual unsigned int getSize(void) const override;
      virtual enum DataType getEnumType(int isSigned = -1) const override;
      virtual unsigned int getElementsNb(void) const override { return elements_nb; }
      virtual unsigned int getElementSize(unsigned int /*index*/) const override;
      virtual Type const* getElementType(unsigned int /*index*/) const override;
      virtual enum DataType getElementEnumType(unsigned int /*index*/, int isSigned = -1) const override;
      virtual unsigned int getGlobalIdx(std::vector<unsigned int> const& elementIds, unsigned position = 0u) const override;
      virtual void getGlobalOffset(BuildUtil *bu, Decoration const& decoration, Value *offset, std::vector<Value *> ids, unsigned position = 0u) const override;
      virtual std::vector<unsigned int> getPaddings() const override;

      spv::Id component_type_id;
      Type* component_type;
      spv::Id elements_nb_id;
      unsigned elements_nb;
   };
   class TypePointer : public Type {
   public:
      TypePointer(unsigned int numWords, unsigned int firstWord,
                  const char *const binary, bool &didSucceed, unsigned int psize,
                  std::unordered_map<spv::Id, Type*> const& types);
      virtual ~TypePointer() {}
      virtual bool isBasicType() const override { return true; }
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override;
      virtual unsigned int getSize(void) const override { return size / 8u; }
      virtual enum DataType getEnumType(int isSigned = -1) const override;
      enum SpirvFile getStorageFile() const { return Converter::getStorageFile(storage); }
      Type* getPointedType() const { return type; }
      virtual void getGlobalOffset(BuildUtil *bu, Decoration const& decoration, Value *offset, std::vector<Value *> ids, unsigned position = 0u) const override;

      spv::StorageClass storage;
      spv::Id type_id;
      Type* type;
      unsigned int size;
   };
   class TypeFunction : public Type {
   public:
      TypeFunction(unsigned int numWords, unsigned int firstWord,
                   const char *const binary, bool &didSucceed);
      virtual ~TypeFunction() {}
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override { assert(false); return std::vector<Value *>(); }

      spv::Id type;
      std::vector<spv::Id> params;
   };
   class TypeSampler : public Type {
   public:
      TypeSampler(unsigned int numWords, unsigned int firstWord,
                  const char *const binary, bool &didSucceed);
      virtual ~TypeSampler() {}
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override { assert(false); return std::vector<Value *>(); }

      spv::Id id;
   };
   class TypeImage : public Type {
   public:
      TypeImage(unsigned int numWords, unsigned int firstWord,
                const char *const binary, bool &didSucceed);
      virtual ~TypeImage() {}
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override { assert(false); return std::vector<Value *>(); }

      spv::Id id;
      spv::Id sampled_type;
      spv::Dim dim;
      word depth;
      word arrayed;
      word ms;
      word sampled;
      spv::ImageFormat format;
      spv::AccessQualifier access;
   };
   class TypeSampledImage : public Type {
   public:
      TypeSampledImage(unsigned int numWords, unsigned int firstWord,
                       const char *const binary, bool &didSucceed);
      virtual ~TypeSampledImage() {}
      virtual std::vector<Value *> generateNullConstant(Converter &conv) const override { assert(false); return std::vector<Value *>(); }
      spv::Id getImageType() const { return image_type; }

      spv::Id id;
      spv::Id image_type;
   };
   struct Sampler {
      TypeSampler const* type;
      spv::SamplerAddressingMode addressingMode;
      bool normalizedCoords;
      spv::SamplerFilterMode filterMode;
   };
   struct SampledImage {
      TypeSampledImage const* type;
      Value* image;
      Sampler sampler;
   };
   struct FunctionData {
      Function* caller;
      FlowInstruction* callInsn;
      FunctionData(Function* _caller, FlowInstruction* _callInsn)
         : caller(_caller), callInsn(_callInsn) {}
   };

   Converter(Program *, struct nv50_ir_prog_info *info);
   ~Converter();

   bool run();

private:
   bool convertInstruction(spv::Op opcode, unsigned int numWords,
                           unsigned int firstWord);
   bool convertEntryPoint(unsigned int numWords, unsigned int firstWord);
   bool convertDecorate(unsigned int numWords, unsigned int firstWord,
                        bool hasMember = false);
   template<typename T> bool convertType(unsigned int numWords,
                                         unsigned int firstWord);
   Symbol * createSymbol(SpirvFile file, DataType type, unsigned int size, unsigned int offset) {
      Symbol *base_symbol = baseSymbols[file];
      Symbol *sym = new_Symbol(prog, base_symbol->reg.file, base_symbol->reg.fileIndex);
      sym->reg.type = type;
      sym->reg.size = size;

      // TODO(pmoreau): This is a hack to get the proper offset on Tesla
      if (file == SpirvFile::INPUT)
         offset += info->prop.cp.inputOffset;

      sym->setAddress(base_symbol, offset);

      return sym;
   }
   nv50_ir::operation convertOp(spv::Op op);
   nv50_ir::CondCode convertCc(spv::Op op);
   bool loadBuiltin(spv::Id dstId, Type const* dstType, Words const& decLiterals, spv::MemoryAccessMask access = spv::MemoryAccessMask::MaskNone);
   bool convertOpenCLInstruction(spv::Id resId, Type const* type, uint32_t op, uint32_t firstWord, uint32_t numWords);
   int getSubOp(spv::Op opcode) const;
   static enum SpirvFile getStorageFile(spv::StorageClass storage);
   static unsigned int getFirstBasicElementSize(Type const* type);
   static enum DataType getFirstBasicElementEnumType(Type const* type);
   static TexTarget getTexTarget(TypeImage const* type);
   static TexInstruction::ImgFormatDesc const* getImageFormat(spv::ImageFormat format);

   Value * acquire(SpirvFile dstFile, Type const* type);
   Value *acquire(SpirvFile file, spv::Id id, Type const* type);
   unsigned load(SpirvFile dstFile, SpirvFile srcFile, spv::Id id, PValue const& ptr, unsigned int offset, Type const* type, spv::MemoryAccessMask access = spv::MemoryAccessMask::MaskNone, uint32_t alignment = 0u);
   void store(SpirvFile dstFile, PValue const& ptr, unsigned int offset, Value *value, DataType stTy, spv::MemoryAccessMask access, uint32_t alignment);
   void store(SpirvFile dstFile, PValue const& ptr, unsigned int offset, std::vector<PValue> const& values, Type const* type, spv::MemoryAccessMask access = spv::MemoryAccessMask::MaskNone, uint32_t alignment = 0u);

   template<typename T> T getWord(unsigned word_offset) { return spirv::getWord<T>(binary, word_offset); }

   struct nv50_ir_prog_info *info;
   const char *const binary;
   std::unordered_map<spv::Id, std::string> extInstructions;
   spv::AddressingModel addressingModel;
   spv::MemoryModel memoryModel;
   std::unordered_map<spv::Id, EntryPoint> entryPoints;
   std::unordered_map<spv::Id, std::string> names;
   std::unordered_map<spv::Id, Decoration> decorations;
   std::unordered_map<spv::Id, Type *> types;
   std::unordered_map<spv::Id, Function *> functions;
   std::unordered_map<spv::Id, BasicBlock *> blocks;
   std::unordered_map<spv::Id, std::vector<std::pair<std::vector<PValue>, BasicBlock*>>> phiNodes;
   std::unordered_map<nv50_ir::Instruction*, spv::Id> phiMapping;
   std::unordered_map<spv::Id, std::unordered_map<uint32_t, std::pair<spv::Id, spv::Id>>> phiToMatch;
   std::unordered_map<spv::Id, Sampler> samplers;
   std::unordered_map<spv::Id, SampledImage> sampledImages;

   ValueMap spvValues;

   std::unordered_map<SpirvFile, Symbol *> baseSymbols;
   spv::Id currentFuncId;
   uint32_t inputOffset; // XXX maybe better to have a separate DataArray for input, keeping track

   std::unordered_map<spv::Id, std::vector<FlowInstruction*>> branchesToMatch;
   std::unordered_map<spv::Id, std::vector<FunctionData>> functionsToMatch;
};

class GetOutOfSSA : public Pass
{
public:
   void setData(std::unordered_map<spv::Id, std::vector<std::pair<std::vector<Converter::PValue>, BasicBlock*>>>* nodes, std::unordered_map<nv50_ir::Instruction*, spv::Id>* mapping, Converter::ValueMap* values) {
      phiNodes = nodes;
      phiMapping = mapping;
      spvValues = values;
   }

private:
   virtual bool visit(BasicBlock *);
   virtual bool visit(Function *);
   bool handlePhi(Instruction *);

   std::unordered_map<spv::Id, std::vector<std::pair<std::vector<Converter::PValue>, BasicBlock*>>>* phiNodes;
   std::unordered_map<nv50_ir::Instruction*, spv::Id>* phiMapping;
   Converter::ValueMap* spvValues;

protected:
   BuildUtil bld;
};

bool
GetOutOfSSA::visit(BasicBlock *bb)
{
   Instruction *next;
   for (Instruction *i = bb->getPhi(); i && i != bb->getEntry(); i = next) {
      next = i->next;
      if (!handlePhi(i))
         return false;
   }
   return true;
}

bool
GetOutOfSSA::visit(Function *func)
{
   bld.setProgram(func->getProgram());
   return true;
}

bool
GetOutOfSSA::handlePhi(Instruction *insn)
{
   auto searchId = phiMapping->find(insn);
   if (searchId == phiMapping->end()) {
      _debug_printf("Couldn't find id linked to phi insn:\n\t");
      insn->print();
      return false;
   }
   auto searchData = phiNodes->find(searchId->second);
   if (searchData == phiNodes->end()) {
      _debug_printf("Couldn't find phi node with id %u\n", searchId->second);
      return false;
   }

   auto& data = searchData->second;
   auto pairs = std::vector<std::pair<std::vector<Converter::PValue>, BasicBlock*>>();
   pairs.reserve(data.size());
   for (Graph::EdgeIterator it = insn->bb->cfg.incident(); !it.end(); it.next()) {
      BasicBlock *obb = BasicBlock::get(it.getNode());
      for (auto& pair : data) {
         if (pair.second != obb)
            continue;

         pairs.push_back(pair);
         break;
      }
   }
   auto searchValue = spvValues->find(searchId->second);
   if (searchValue == spvValues->end()) {
      _debug_printf("Couldn't find SpirVValue for phi node with id %u\n", searchId->second);
      return false;
   }

   for (auto& pair : pairs) {
      if (pair.first.size() > 1u)
         _debug_printf("Multiple var for same phi node aren't really supported\n");
      auto bbExit = pair.second->getExit();
      if (bbExit == nullptr) {
         _debug_printf("BB.exit == nullptr; this is unexpected, things will go wrong!\n");
         return false;
      }
      bld.setPosition(bbExit, !(bbExit->op == OP_BRA || bbExit->op == OP_EXIT));
      bld.mkMov(searchValue->second.value[0].value, pair.first[0].value, searchValue->second.type->getEnumType());
   }

   delete_Instruction(bld.getProgram(), insn);

   return true;
}

Converter::TypeVoid::TypeVoid(unsigned int numWords, unsigned int firstWord,
                              const char *const binary, bool &didSucceed) : Type(spv::Op::OpTypeVoid)
{
   if (numWords != 2u) {
      _debug_printf("OpTypeVoid expects 2 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   didSucceed = true;
}

Converter::TypeBool::TypeBool(unsigned int numWords, unsigned int firstWord,
                                const char *const binary, bool &didSucceed) : Type(spv::Op::OpTypeBool)
{
   if (numWords != 2u) {
      _debug_printf("OpTypeBool expects 2 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   alignment = 1u;
   didSucceed = true;
}

std::vector<ImmediateValue *>
Converter::TypeBool::generateConstant(const Words &words, Converter &conv, unsigned int& position) const
{
   return { conv.mkImm(words[position++]) }; // XXX unsure
}

std::vector<Value *>
Converter::TypeBool::generateNullConstant(Converter &conv) const
{
   return { conv.mkImm(0u) };
}

enum DataType
Converter::TypeBool::getEnumType(int /*isSigned*/) const
{
   return DataType::TYPE_NONE;
}

Converter::TypeInt::TypeInt(unsigned int numWords, unsigned int firstWord,
                                const char *const binary, bool &didSucceed) : Type(spv::Op::OpTypeInt)
{
   if (numWords != 4u) {
      _debug_printf("OpTypeInt expects 4 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   width = spirv::getWord<unsigned>(binary, firstWord + 1u);
   signedness = spirv::getWord<unsigned>(binary, firstWord + 2u);
   alignment = width / 8u;
   didSucceed = true;
}

std::vector<ImmediateValue *>
Converter::TypeInt::generateConstant(const Words &words, Converter &conv, unsigned int& position) const
{
   if (words.size() > position + 2u) {
      _debug_printf("Int values can't be more than 64-bits wide\n");
      return std::vector<ImmediateValue *>();
   }

   ImmediateValue *imm = nullptr;
   if (width != 64u) {
      imm = conv.mkImm(words[position]);
      ++position;
   } else {
      imm = conv.mkImm((static_cast<unsigned long>(words[position + 1u]) << 32u) +
                        static_cast<unsigned long>(words[position]));
      position += 2u;
   }
   imm->reg.type = getEnumType();
   return { imm };
}

std::vector<Value *>
Converter::TypeInt::generateNullConstant(Converter &conv) const
{
   return { (width == 64u) ? conv.mkImm(0ul) : conv.mkImm(0u) };
}

enum DataType
Converter::TypeInt::getEnumType(int isSigned) const
{
   if (isSigned == 1 || (isSigned == -1 && signedness == 1u)) {
      if (width == 8u)
         return DataType::TYPE_S8;
      else if (width == 16u)
         return DataType::TYPE_S16;
      else if (width == 32u)
         return DataType::TYPE_S32;
      else if (width == 64u)
         return DataType::TYPE_S64;
      else {
         _debug_printf("TypeInt has a non valid width of %u bits\n", width);
         assert(false);
         return DataType::TYPE_NONE;
      }
   }
   if (isSigned == 0 || (isSigned == -1 && signedness == 0u)) {
      if (width == 8u)
         return DataType::TYPE_U8;
      else if (width == 16u)
         return DataType::TYPE_U16;
      else if (width == 32u)
         return DataType::TYPE_U32;
      else if (width == 64u)
         return DataType::TYPE_U64;
      else {
         _debug_printf("TypeInt has a non valid width of %u bits\n", width);
         assert(false);
         return DataType::TYPE_NONE;
      }
   }
}

Converter::TypeFloat::TypeFloat(unsigned int numWords, unsigned int firstWord,
                                const char *const binary, bool &didSucceed) : Type(spv::Op::OpTypeFloat)
{
   if (numWords != 3u) {
      _debug_printf("OpTypeFloat expects 3 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   width = spirv::getWord<unsigned>(binary, firstWord + 1u);
   alignment = width / 8u;
   didSucceed = true;
}

std::vector<ImmediateValue *>
Converter::TypeFloat::generateConstant(const Words &words, Converter &conv, unsigned int& position) const
{
   if (words.size() > position + 2u) {
      _debug_printf("Floating point values can't be more than 64-bits wide\n");
      return std::vector<ImmediateValue *>();
   }

   ImmediateValue *imm = nullptr;
   if (width != 64u) {
      imm =  conv.mkImm(words[position]);
      ++position;
   } else {
      imm = conv.mkImm((static_cast<unsigned long>(words[position + 1u]) << 32u) +
                        static_cast<unsigned long>(words[position]));
      position += 2u;
   }
   imm->reg.type = getEnumType();
   return { imm };
}

std::vector<Value *>
Converter::TypeFloat::generateNullConstant(Converter &conv) const
{
   return { (width == 64u) ? conv.mkImm(0.0) : conv.mkImm(0.0f) };
}

enum DataType
Converter::TypeFloat::getEnumType(int /*isSigned*/) const
{
   if (width == 16u)
      return DataType::TYPE_F16;
   else if (width == 32u)
      return DataType::TYPE_F32;
   else if (width == 64u)
      return DataType::TYPE_F64;
   else {
      _debug_printf("TypeFloat has a non valid width of %u bits\n", width);
      assert(false);
      return DataType::TYPE_NONE;
   }
}

Converter::TypeStruct::TypeStruct(unsigned int numWords, unsigned int firstWord,
                                  const char *const binary, bool &didSucceed,
                                  std::unordered_map<spv::Id, Type*> const& types,
                                  std::unordered_map<spv::Id, Decoration> const& decorations) : Type(spv::Op::OpTypeStruct)
{
   if (numWords < 2u) {
      _debug_printf("OpTypeStruct expects at least 2 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   size = 0u;
   member_ids.reserve(numWords - 2u);
   member_types.reserve(numWords - 2u);
   auto largest_alignment = 0u;

   bool is_packed = false;
   const auto &decos = decorations.find(id);
   if (decos != decorations.end())
      is_packed = decos->second.find(spv::Decoration::CPacked) != decos->second.end();

   for (unsigned int i = 1u; i < numWords - 1u; ++i) {
      const auto member_id = spirv::getWord<spv::Id>(binary, firstWord + i);
      auto search = types.find(member_id);
      if (search == types.end()) {
         _debug_printf("Couldn't find the type %u associated to TypeStruct %u\n", member_id, id);
         didSucceed = false;
         return;
      }
      member_ids.push_back(member_id);
      member_types.push_back(search->second);

      const auto member_size = search->second->getSize();
      const auto member_alignment = is_packed ? 1u : search->second->getAlignment();
      largest_alignment = std::max(largest_alignment, member_alignment);
      const auto padding = (-size) & (member_alignment - 1u);
      size += padding + member_size;

      if (search->second->isCompooundType()) {
         auto paddings = search->second->getPaddings();
         paddings[0] += padding;
         member_paddings.insert(member_paddings.end(), paddings.begin(), paddings.end());
      } else {
         member_paddings.push_back(padding);
      }
   }
   size += (-size) & (largest_alignment - 1u);
   alignment = largest_alignment;

   didSucceed = true;
}

std::vector<ImmediateValue *>
Converter::TypeStruct::generateConstant(const Words &words, Converter &conv, unsigned int& position) const
{
   std::vector<ImmediateValue *> imms;
   for (const Type *member_type : member_types) {
      const auto member_constant = member_type->generateConstant(words, conv, position);
      imms.insert(imms.end(), member_constant.begin(), member_constant.end());
   }
   return imms;
}

std::vector<Value *>
Converter::TypeStruct::generateNullConstant(Converter &conv) const
{
   std::vector<Value *> null_constant;
   for (const Type *member_type : member_types) {
      const auto member_constant = member_type->generateNullConstant(conv);
      null_constant.insert(null_constant.end(), member_constant.begin(), member_constant.end());
   }
   return null_constant;
}

enum DataType
Converter::TypeStruct::getEnumType(int /*isSigned*/) const
{
   return DataType::TYPE_NONE;
}

unsigned int
Converter::TypeStruct::getElementSize(unsigned int index) const
{
   assert(index < member_types.size());
   assert(member_types[index] != nullptr);
   return member_types[index]->getSize();
}

Converter::Type const*
Converter::TypeStruct::getElementType(unsigned int index) const
{
   assert(index < member_types.size());
   assert(member_types[index] != nullptr);
   return member_types[index];
}

enum DataType
Converter::TypeStruct::getElementEnumType(unsigned int index, int isSigned) const
{
   assert(index < member_types.size());
   assert(member_types[index] != nullptr);
   return member_types[index]->getEnumType(isSigned);
}

unsigned int
Converter::TypeStruct::getGlobalIdx(std::vector<unsigned int> const& elementIds, unsigned position) const
{
   assert(position == elementIds.size() - 1u);
   return elementIds[position];
}

void
Converter::TypeStruct::getGlobalOffset(BuildUtil *bu, Decoration const& decoration, Value *offset, std::vector<Value *> ids, unsigned position) const
{
   assert(position < ids.size());

   const auto imm = ids[position];
   uint32_t struct_off = 0u;
   for (int i = 0; i < imm->reg.data.u32; ++i)
      struct_off += member_types[i]->getSize();
   auto res = bu->getScratch(offset->reg.size);
   if (offset->reg.type == TYPE_U64)
      bu->loadImm(res, static_cast<unsigned long>(struct_off));
   else
      bu->loadImm(res, struct_off);
   bu->mkOp2(OP_ADD, offset->reg.type, offset, offset, res);

   if (position + 1u < ids.size()) {
      _debug_printf("Trying to dereference basic types\n");
      assert(false);
   }
}

Converter::TypeVector::TypeVector(unsigned int numWords, unsigned int firstWord,
                                  const char *const binary, bool &didSucceed,
                                  std::unordered_map<spv::Id, Type*> const& types) : Type(spv::Op::OpTypeVector)
{
   if (numWords != 4u) {
      _debug_printf("OpTypeVector expects 4 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   component_type_id = spirv::getWord<spv::Id>(binary, firstWord + 1u);
   auto search = types.find(component_type_id);
   if (search == types.end()) {
      _debug_printf("Couldn't find the type associated to TypeVector %u\n", id);
      didSucceed = false;
      return;
   }
   component_type = search->second;
   elements_nb = spirv::getWord<unsigned>(binary, firstWord + 2u);
   alignment = component_type->getAlignment();
   didSucceed = true;
}

std::vector<ImmediateValue *>
Converter::TypeVector::generateConstant(const Words &words, Converter &conv, unsigned int& position) const
{
   std::vector<ImmediateValue *> imms_constant;
   const auto member_constant = component_type->generateConstant(words, conv, position);
   for (unsigned int i = 0u; i < elements_nb; ++i)
      imms_constant.insert(imms_constant.end(), member_constant.begin(), member_constant.end());
   return imms_constant;
}

std::vector<Value *>
Converter::TypeVector::generateNullConstant(Converter &conv) const
{
   std::vector<Value *> null_constant;
   const auto member_constant = component_type->generateNullConstant(conv);
   for (unsigned int i = 0u; i < elements_nb; ++i)
      null_constant.insert(null_constant.end(), member_constant.begin(), member_constant.end());
   return null_constant;
}

unsigned int
Converter::TypeVector::getSize(void) const
{
   assert(component_type != nullptr);
   return component_type->getSize() * elements_nb;
}

Converter::Type const*
Converter::TypeVector::getElementType(unsigned int /*index*/) const
{
   assert(component_type != nullptr);
   return component_type;
}

enum DataType
Converter::TypeVector::getEnumType(int /*isSigned*/) const
{
   return DataType::TYPE_NONE;
}

unsigned int
Converter::TypeVector::getElementSize(unsigned int /*index*/) const
{
   assert(component_type != nullptr);
   return component_type->getSize();
}

enum DataType
Converter::TypeVector::getElementEnumType(unsigned int /*index*/, int isSigned) const
{
   assert(component_type != nullptr);
   return component_type->getEnumType(isSigned);
}


unsigned int
Converter::TypeVector::getGlobalIdx(std::vector<unsigned int> const& elementIds, unsigned position) const
{
   assert(position == elementIds.size() - 1u);
   return elementIds[position];
}

void
Converter::TypeVector::getGlobalOffset(BuildUtil *bu, Decoration const& decoration, Value *offset, std::vector<Value *> ids, unsigned position) const
{
   assert(component_type != nullptr && position < ids.size());

   auto res = bu->getScratch(offset->reg.size);
   if (offset->reg.type == TYPE_U64)
      bu->loadImm(res, static_cast<unsigned long>(component_type->getSize()));
   else
      bu->loadImm(res, component_type->getSize());
   bu->mkOp3(OP_MAD, offset->reg.type, offset, ids[position], res, offset);

   if (position + 1u < ids.size()) {
      _debug_printf("Trying to dereference basic types\n");
      assert(false);
   }
}

std::vector<unsigned int>
Converter::TypeVector::getPaddings() const
{
   std::vector<unsigned int> paddings;
   const auto element_paddings = component_type->getPaddings();
   for (unsigned int i = 0u; i < elements_nb; ++i)
      paddings.insert(paddings.end(), element_paddings.cbegin(), element_paddings.cend());
   return paddings;
}

Converter::TypeArray::TypeArray(unsigned int numWords, unsigned int firstWord,
                                  const char *const binary, bool &didSucceed,
                                  std::unordered_map<spv::Id, Type*> const& types,
                                  const ValueMap &m) : Type(spv::Op::OpTypeArray)
{
   if (numWords != 4u) {
      _debug_printf("OpTypeArray expects 4 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   component_type_id = spirv::getWord<spv::Id>(binary, firstWord + 1u);
   auto search = types.find(component_type_id);
   if (search == types.end()) {
      _debug_printf("Couldn't find the type associated to TypeArray %u\n", id);
      didSucceed = false;
      return;
   }
   component_type = search->second;
   elements_nb_id = spirv::getWord<spv::Id>(binary, firstWord + 2u);
   auto searchElemNb = m.find(elements_nb_id);
   assert(searchElemNb != m.end() && searchElemNb->second.storageFile == SpirvFile::IMMEDIATE);
   elements_nb = searchElemNb->second.value.front().value->asImm()->reg.data.u32;
   alignment = component_type->getAlignment();
   didSucceed = true;
}

std::vector<Value *>
Converter::TypeArray::generateNullConstant(Converter &conv) const
{
   std::vector<Value *> null_constant;
   const auto member_constant = component_type->generateNullConstant(conv);
   for (unsigned int i = 0u; i < elements_nb; ++i)
      null_constant.insert(null_constant.end(), member_constant.begin(), member_constant.end());
   return null_constant;
}

Converter::Type const*
Converter::TypeArray::getElementType(unsigned int /*index*/) const
{
   assert(component_type != nullptr);
   return component_type;
}

unsigned int
Converter::TypeArray::getSize(void) const
{
   assert(component_type != nullptr);
   assert(elements_nb != 0u);
   return component_type->getSize() * elements_nb;
}

enum DataType
Converter::TypeArray::getEnumType(int /*isSigned*/) const
{
   return DataType::TYPE_NONE;
}

unsigned int
Converter::TypeArray::getElementSize(unsigned int /*index*/) const
{
   assert(component_type != nullptr);
   return component_type->getSize();
}

enum DataType
Converter::TypeArray::getElementEnumType(unsigned int /*index*/, int isSigned) const
{
   assert(component_type != nullptr);
   return component_type->getEnumType(isSigned);
}


unsigned int
Converter::TypeArray::getGlobalIdx(std::vector<unsigned int> const& elementIds, unsigned position) const
{
   assert(position == elementIds.size() - 1u);
   return elementIds[position];
}

void
Converter::TypeArray::getGlobalOffset(BuildUtil *bu, Decoration const& decoration, Value *offset, std::vector<Value *> ids, unsigned position) const
{
   assert(component_type != nullptr && position < ids.size());

   auto res = bu->getScratch(offset->reg.size);
   if (offset->reg.type == TYPE_U64)
      bu->loadImm(res, static_cast<unsigned long>(component_type->getSize()));
   else
      bu->loadImm(res, component_type->getSize());
   bu->mkOp3(OP_MAD, offset->reg.type, offset, ids[position], res, offset);

   component_type->getGlobalOffset(bu, decoration, offset, ids, position + 1u);
}

std::vector<unsigned int>
Converter::TypeArray::getPaddings() const
{
   std::vector<unsigned int> paddings;
   const auto element_paddings = component_type->getPaddings();
   for (unsigned int i = 0u; i < elements_nb; ++i)
      paddings.insert(paddings.end(), element_paddings.begin(), element_paddings.end());
   return paddings;
}

Converter::TypePointer::TypePointer(unsigned int numWords, unsigned int firstWord,
                                    const char *const binary, bool &didSucceed,
                                    unsigned int psize,
                                    std::unordered_map<spv::Id, Type*> const& types) : Type(spv::Op::OpTypePointer)
{
   if (numWords != 4u) {
      _debug_printf("OpTypePointer expects 4 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   storage = spirv::getWord<spv::StorageClass>(binary, firstWord + 1u);
   type_id = spirv::getWord<spv::Id>(binary, firstWord + 2u);
   auto search = types.find(type_id);
   if (search == types.end()) {
      _debug_printf("Couldn't find the type associated to TypePointer %u\n", id);
      didSucceed = false;
      return;
   }
   type = search->second;
   didSucceed = true;
   size = psize;
   alignment = size / 8u;
}

std::vector<Value *>
Converter::TypePointer::generateNullConstant(Converter &conv) const
{
   return { (size == 32u) ? conv.mkImm(0u) : conv.mkImm(0ul) };
}

enum DataType
Converter::TypePointer::getEnumType(int /*isSigned*/) const
{
   if (size == 32u)
      return DataType::TYPE_U32;
   else if (size == 64u)
      return DataType::TYPE_U64;
   else {
      _debug_printf("TypePointer has a non valid size of %u bits\n", size);
      assert(false);
      return DataType::TYPE_NONE;
   }
}

void
Converter::TypePointer::getGlobalOffset(BuildUtil *bu, Decoration const& decoration, Value *offset, std::vector<Value *> ids, unsigned position) const
{
   assert(position < ids.size());

   if (storage != spv::StorageClass::Function) {
      unsigned int type_size = type->getSize();

      auto search_alignment = decoration.find(spv::Decoration::Alignment);
      if (search_alignment != decoration.end())
         type_size += (-type_size) & (search_alignment->second[0][0] - 1u);

      Value *tmp = bu->getScratch(offset->reg.size);
      if (offset->reg.type == TYPE_U64)
         bu->loadImm(tmp, static_cast<unsigned long>(type_size));
      else
         bu->loadImm(tmp, type_size);
      bu->mkOp3(OP_MAD, offset->reg.type, offset, tmp, ids[position], offset);
   } else {
      assert(ids[position]->asImm() != nullptr && ids[position]->asImm()->reg.data.u64 == 0ul);
   }

   if (position + 1u < ids.size())
      type->getGlobalOffset(bu, decoration, offset, ids, position + 1u);
}

Converter::TypeFunction::TypeFunction(unsigned int numWords, unsigned int firstWord,
                                      const char *const binary, bool &didSucceed) : Type(spv::Op::OpTypeFunction)
{
   if (numWords < 3u) {
      _debug_printf("OpTypeFunction expects at least 3 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   type = spirv::getWord<spv::Id>(binary, firstWord + 1u);
   for (unsigned int i = 2u; i < numWords; ++i)
      params.push_back(spirv::getWord<spv::Id>(binary, firstWord + i));
   alignment = 0u;
   didSucceed = true;
}

Converter::TypeSampler::TypeSampler(unsigned int numWords, unsigned int firstWord,
                                    const char *const binary, bool &didSucceed) : Type(spv::Op::OpTypeSampler)
{
   if (numWords != 2u) {
      _debug_printf("OpTypeSampler expects 2 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   alignment = 0u;
   didSucceed = true;
}

Converter::TypeSampledImage::TypeSampledImage(unsigned int numWords, unsigned int firstWord,
                                              const char *const binary, bool &didSucceed) : Type(spv::Op::OpTypeSampledImage)
{
   if (numWords != 3u) {
      _debug_printf("OpTypeSampledImage expects 3 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   image_type = spirv::getWord<spv::Id>(binary, firstWord + 1);
   alignment = 0u;
   didSucceed = true;
}

Converter::TypeImage::TypeImage(unsigned int numWords, unsigned int firstWord,
                                const char *const binary, bool &didSucceed) : Type(spv::Op::OpTypeImage)
{
   if (numWords < 9u) {
      _debug_printf("OpTypeImage expects at least 9 operands but got %u\n", numWords);
      didSucceed = false;
      return;
   }

   id = spirv::getWord<spv::Id>(binary, firstWord);
   sampled_type = spirv::getWord<spv::Id>(binary, firstWord + 1u);
   dim = spirv::getWord<spv::Dim>(binary, firstWord + 2u);
   depth = spirv::getWord<unsigned>(binary, firstWord + 3u);
   arrayed = spirv::getWord<unsigned>(binary, firstWord + 4u);
   ms = spirv::getWord<unsigned>(binary, firstWord + 5u);
   sampled = spirv::getWord<unsigned>(binary, firstWord + 6u);
   format = spirv::getWord<spv::ImageFormat>(binary, firstWord + 7u);
   if (numWords == 10)
      access = spirv::getWord<spv::AccessQualifier>(binary, firstWord + 8u);
   alignment = 0u;
   didSucceed = true;
}

Value *
Converter::acquire(SpirvFile dstFile, Type const* type)
{
   assert(type != nullptr);

   if (dstFile == SpirvFile::TEMPORARY) {
      Value *res = nullptr; // FIXME still meh
      if (getFunction()) {
         res = getScratch(std::max(4u, type->getSize()));
         res->reg.type = type->getEnumType();
      }
      return res;
   }

   return createSymbol(dstFile, getFirstBasicElementEnumType(type), std::max(4u, getFirstBasicElementSize(type)), 0u);
}

Value *
Converter::acquire(SpirvFile file, spv::Id id, Type const* type)
{
   assert(type != nullptr);

   auto values = std::vector<PValue>();
   Value *res = nullptr;

   auto save_to_share = file == SpirvFile::SHARED;

   const Type *processed_type = type;
   std::stack<const Type *> types;
   auto ptr_type = reinterpret_cast<const TypePointer *>(type);
   if (type->getType() == spv::Op::OpTypePointer && ptr_type->getStorageFile() == SpirvFile::TEMPORARY)
      processed_type = ptr_type->getPointedType();
   types.push(processed_type);

   while (!types.empty()) {
      const Type *currentType = types.top();
      types.pop();

      if (currentType->isCompooundType()) {
         for (unsigned int i = currentType->getElementsNb(); i > 0u; --i)
            types.push(currentType->getElementType(i - 1u));
         continue;
      }

      res = acquire(file, currentType);
      if (res->reg.file == FILE_GPR)
         values.push_back(res);
      else
         values.emplace_back(res->asSym(), nullptr);
      if (save_to_share)
         info->bin.smemSize += currentType->getSize();
   }

   spvValues.emplace(id, SpirVValue{ file, type, values, processed_type->getPaddings() });

   return res;
}

// TODO coalescing should occur on alignment size
unsigned
Converter::load(SpirvFile dstFile, SpirvFile srcFile, spv::Id id, PValue const& ptr, unsigned int offset, Type const* type, spv::MemoryAccessMask access, uint32_t alignment)
{
   assert(type != nullptr);

   auto values = std::vector<PValue>();

   const auto hasLoadAlignment = hasFlag(access, spv::MemoryAccessShift::Aligned);
   unsigned processedAlignment = 0u;
   auto localOffset = offset;
   std::stack<Type const*> stack;
   stack.push(type);
   Value *coalescedLoad = nullptr;
   while (!stack.empty()) {
      unsigned deltaOffset = 0u;
      auto currentType = stack.top();
      stack.pop();
      if (!currentType->isCompooundType()) {
         if (hasLoadAlignment && processedAlignment >= alignment) {
            const auto mod = processedAlignment % alignment;
            if (mod)
               deltaOffset += alignment - mod;
            processedAlignment = 0u;
         }

         const auto elemSize = currentType->getSize();
         auto mod = (localOffset + deltaOffset) % elemSize;
         if (mod)
            deltaOffset += elemSize - mod;
         localOffset += deltaOffset;
         processedAlignment += deltaOffset;

         const auto size = elemSize;
         mod = localOffset % size;

         Value *res = nullptr;
         // We coalesce as many elements as possible to get a load of at least
         // 32 bits, and use shifts and ANDs to properly split the coalesced
         // results.
         if (mod == 0u) {
            // TODO make use of MemoryAccess::Nontemporal
            const auto enumType = typeOfSize(size);
            const auto gprSize = std::max(4u, size);
            if (srcFile == SpirvFile::IMMEDIATE || (ptr.indirect != nullptr && ptr.indirect->reg.file == FILE_IMMEDIATE)) {
               res = getScratch(gprSize);
               res->reg.type = enumType;
               mkMov(res, ptr.indirect, enumType);
            } else {
               Symbol *sym = ptr.symbol;
               if (sym == nullptr)
                  sym = createSymbol(srcFile, enumType, size, localOffset);

               res = getScratch(gprSize);
               res->reg.type = enumType;
               Instruction* insn = nullptr;
               insn = mkLoad(enumType, res, sym, ptr.indirect);
               if (hasFlag(access, spv::MemoryAccessShift::Volatile))
                  insn->fixed = 1;
            }

            if (elemSize != size)
               coalescedLoad = res;
         }

         if (elemSize != size) {
            res = getScratch();
            mkMov(res, coalescedLoad, TYPE_U32);
            if (mod != 0u) {
               Value *imm = mkImm(mod * (elemSize * 8u));
               Value *immVal = getScratch();
               mkMov(immVal, imm, TYPE_U32);
               mkOp2(OP_SHR, TYPE_U32, res, res, immVal); // FIXME sign of shift op
            }
            Value *mask = (elemSize == 1u) ? mkImm(0xffu) : mkImm(0xffffu);
            Value *maskVal = getScratch();
            mkMov(maskVal, mask, TYPE_U32);
            mkOp2(OP_AND, TYPE_U32, res, res, maskVal);
         }

         localOffset += elemSize;
         processedAlignment += elemSize;
         if (res->reg.file == FILE_GPR)
            values.push_back(res);
         else
            values.emplace_back(res->asSym(), nullptr);
      } else {
         for (unsigned int i = currentType->getElementsNb(); i != 0u; --i)
            stack.push(currentType->getElementType(i - 1u));
      }
   }

   spvValues.emplace(id, SpirVValue{ dstFile, type, values, type->getPaddings() });

   return localOffset - offset;
}

// TODO use access
void
Converter::store(SpirvFile dstFile, PValue const& ptr, unsigned int offset, Value *value, DataType stTy, spv::MemoryAccessMask access, uint32_t alignment)
{
   assert(value != nullptr);

   Value *realValue = value;
   if (value->reg.file == FILE_IMMEDIATE) {
      realValue = getScratch(value->reg.size);
      mkMov(realValue, value, value->reg.type);
   }

   if (dstFile == SpirvFile::TEMPORARY) {
      mkMov(ptr.indirect, realValue, realValue->reg.type);
      return;
   }

   Symbol *sym = ptr.symbol;
   if (sym == nullptr)
      sym = createSymbol(dstFile, realValue->reg.type, realValue->reg.size, offset);

   // TODO(pmoreau): This is a hack to get the proper offset on Tesla
   Value *tmp = nullptr;
   if (info->target >= 0xc0)
      tmp = ptr.indirect;
   else
      tmp = mkOp2v(OP_ADD, ptr.indirect->reg.type, getScratch(ptr.indirect->reg.size), ptr.indirect, loadImm(NULL, offset));
   Instruction* insn = mkStore(OP_STORE, stTy, sym, tmp, realValue);
   if (hasFlag(access, spv::MemoryAccessShift::Volatile))
      insn->fixed = 1;
}

void
Converter::store(SpirvFile dstFile, PValue const& ptr, unsigned int offset, std::vector<PValue> const& values, Type const* type, spv::MemoryAccessMask access, uint32_t alignment)
{
   assert(type != nullptr);

   const auto hasStoreAlignment = hasFlag(access, spv::MemoryAccessShift::Aligned);
   unsigned processedAlignment = 0u;
   auto localOffset = offset;
   std::stack<Type const*> stack;
   stack.push(type);
   Value *coalescedStore = nullptr;
   unsigned int c = 0u;
   while (!stack.empty()) {
      unsigned deltaOffset = 0u;
      Value *value = values[c].value;
      auto currentType = stack.top();
      stack.pop();
      if (!currentType->isCompooundType()) {
         if (hasStoreAlignment && processedAlignment >= alignment) {
            const auto mod = processedAlignment % alignment;
            if (mod)
               deltaOffset += alignment - mod;
            processedAlignment = 0u;
         }

         const auto elemSize = currentType->getSize();
         auto mod = (localOffset + deltaOffset) % elemSize;
         if (mod)
            deltaOffset += elemSize - mod;
         localOffset += deltaOffset;
         processedAlignment += deltaOffset;

         const auto size = std::max(4u, elemSize);
         mod = localOffset % size;

         if (elemSize != size) {
            Value *tmp = getScratch();
            if (mod != 0u) {
               Value *imm = mkImm(mod * (elemSize * 8u));
               Value *immVal = getScratch();
               mkMov(immVal, imm, TYPE_U32);
               mkOp2(OP_SHL, TYPE_U32, tmp, value, immVal); // FIXME sign of shift op
            }

            Value *mask = (elemSize == 1u) ? mkImm(0xffu) : mkImm(0xffffu);
            Value *maskVal = getScratch();
            mkMov(maskVal, mask, TYPE_U32);
            mkOp2(OP_AND, TYPE_U32, tmp, (mod == 0u) ? value : tmp, maskVal);

            if (mod == 0u) {
               coalescedStore = getScratch();
               mkMov(coalescedStore, tmp, TYPE_U32);
            } else {
               mkOp2(OP_OR, TYPE_U32, coalescedStore, coalescedStore, tmp);
            }
         }

         // We coalesce as many elements as possible to get a store of at least
         // 32 bits, and use shifts and ORs to properly compose the coalesced
         // value to store.
         if (size == elemSize || mod + 1u == size || stack.empty())
            store(dstFile, ptr, localOffset - mod * elemSize, (size == elemSize) ? value : coalescedStore, (size == elemSize) ? typeOfSize(size) : typeOfSize(elemSize), access, alignment);

         localOffset += elemSize;
         processedAlignment += elemSize;
         ++c;
      } else {
         for (unsigned int i = currentType->getElementsNb(); i != 0u; --i)
            stack.push(currentType->getElementType(i - 1u));
      }
   }
}

Converter::Converter(Program *prog, struct nv50_ir_prog_info *info) : BuildUtil(prog),
   info(info), binary(reinterpret_cast<const char *const>(info->bin.source)),
   extInstructions(), addressingModel(),
   memoryModel(), entryPoints(), decorations(), types(),
   functions(), blocks(), phiNodes(), phiMapping(), phiToMatch(),
   samplers(), sampledImages(), spvValues(), currentFuncId(0u),
   inputOffset(0u), branchesToMatch(), functionsToMatch()
{
   baseSymbols[SpirvFile::TEMPORARY] = new_Symbol(prog, FILE_GPR);
   baseSymbols[SpirvFile::SHARED]    = new_Symbol(prog, FILE_MEMORY_SHARED);
   baseSymbols[SpirvFile::GLOBAL]    = new_Symbol(prog, FILE_MEMORY_GLOBAL, 15);
   baseSymbols[SpirvFile::CONST]     = new_Symbol(prog, FILE_MEMORY_CONST);
   baseSymbols[SpirvFile::PREDICATE] = new_Symbol(prog, FILE_PREDICATE);

   if (info->target >= 0xc0) {
      baseSymbols[SpirvFile::SHARED]->setOffset(info->prop.cp.sharedOffset);
      baseSymbols[SpirvFile::CONST]->setOffset(info->prop.cp.inputOffset);
      baseSymbols[SpirvFile::INPUT] = baseSymbols[SpirvFile::CONST];
   } else {
      baseSymbols[SpirvFile::SHARED]->setOffset(info->prop.cp.inputOffset);
      baseSymbols[SpirvFile::INPUT] = baseSymbols[SpirvFile::SHARED];
   }
}

Converter::~Converter()
{
   for (auto &i : types)
      delete i.second;
}

Converter::SpirvFile
Converter::getStorageFile(spv::StorageClass storage)
{
   switch (storage) {
   case spv::StorageClass::UniformConstant:
      return SpirvFile::CONST;
   case spv::StorageClass::Input:
      return SpirvFile::INPUT; // XXX
   case spv::StorageClass::Workgroup:
      return SpirvFile::SHARED;
   case spv::StorageClass::CrossWorkgroup: // FALLTHROUGH
   case spv::StorageClass::Private:
      return SpirvFile::GLOBAL;
   case spv::StorageClass::Function:
      return SpirvFile::TEMPORARY;
   default:
      _debug_printf("StorageClass %u isn't supported yet\n");
      assert(false);
      return SpirvFile::NONE;
   }
}

unsigned int
Converter::getFirstBasicElementSize(Type const* type)
{
   Type const* currType = type;
   while (!currType->isBasicType())
      currType = currType->getElementType(0u);
   return currType->getSize();
}

enum DataType
Converter::getFirstBasicElementEnumType(Type const* type)
{
   Type const* currType = type;
   while (!currType->isBasicType())
      currType = currType->getElementType(0u);
   return currType->getEnumType();
}

bool
Converter::run()
{
   if (info->dbgFlags)
      _debug_printf("Compiling for nv%02x\n", info->target);

   // TODO try to remove/get around that main function
   BasicBlock *entry = new BasicBlock(prog->main);
   prog->main->setEntry(entry);
   prog->main->setExit(new BasicBlock(prog->main));

   spv::Op opcode;
   unsigned int numOperands = 0u, numWords = info->bin.sourceLength / 4u, i = 5u;
   while (i < numWords) {
      opcode = static_cast<spv::Op>(getWord<unsigned>(i) & spv::OpCodeMask);
      numOperands = getWord<unsigned>(i) >> spv::WordCountShift;

      if (i + (numOperands - 1u) >= numWords) {
         _debug_printf("Opcode %u is missing some operands\n", opcode);
         return false;
      }
      if (!convertInstruction(opcode, numOperands, i + 1u)) {
         _debug_printf("Failed to convert opcode %u\n", opcode);
         prog->print();
         return false;
      }

      i += numOperands;
   }

   for (auto& i : functionsToMatch) {
      auto funcIter = functions.find(i.first);
      if (funcIter == functions.end()) {
         _debug_printf("Unable to find function %u\n", i.first);
         return false;
      }
      auto f = funcIter->second;
      for (auto& j : i.second) {
         j.callInsn->target.fn = f;
         j.caller->call.attach(&f->call, Graph::Edge::TREE);
      }
   }
   functionsToMatch.clear();

   GetOutOfSSA outOfSSAPass;
   outOfSSAPass.setData(&phiNodes, &phiMapping, &spvValues);
   outOfSSAPass.run(prog, true, false);

   return true;
}

template<typename T> bool
Converter::convertType(unsigned int numWords, unsigned int firstWord)
{
   bool didSucceed = false;
   T *type = new T(numWords, firstWord, binary, didSucceed);
   if (!didSucceed) {
      delete type;
      return false;
   }
   types.emplace(type->id, type);

   return true;
}

template<> bool
Converter::convertType<Converter::TypeStruct>(unsigned int numWords, unsigned int firstWord)
{
   bool didSucceed = false;
   auto *type = new TypeStruct(numWords, firstWord, binary, didSucceed,
         types, decorations);
   if (!didSucceed) {
      delete type;
      return false;
   }
   types.emplace(type->id, type);

   return true;
}

template<> bool
Converter::convertType<Converter::TypeVector>(unsigned int numWords, unsigned int firstWord)
{
   bool didSucceed = false;
   auto *type = new TypeVector(numWords, firstWord, binary, didSucceed,
         types);
   if (!didSucceed) {
      delete type;
      return false;
   }
   types.emplace(type->id, type);

   return true;
}

template<> bool
Converter::convertType<Converter::TypeArray>(unsigned int numWords, unsigned int firstWord)
{
   bool didSucceed = false;
   auto *type = new TypeArray(numWords, firstWord, binary, didSucceed,
         types, spvValues);
   if (!didSucceed) {
      delete type;
      return false;
   }
   types.emplace(type->id, type);

   return true;
}

template<> bool
Converter::convertType<Converter::TypePointer>(unsigned int numWords, unsigned int firstWord)
{
   bool didSucceed = false;
   auto *type = new TypePointer(numWords, firstWord, binary, didSucceed,
         (info->target < 0xc0) ? 32u : 64u, types);
   if (!didSucceed) {
      delete type;
      return false;
   }
   types.emplace(type->id, type);

   return true;
}

nv50_ir::operation
Converter::convertOp(spv::Op op)
{
   switch (op) {
   case spv::Op::OpSNegate:
   case spv::Op::OpFNegate:
      return OP_NEG;
   case spv::Op::OpIAdd:
   case spv::Op::OpFAdd:
      return OP_ADD;
   case spv::Op::OpISub:
   case spv::Op::OpFSub:
      return OP_SUB;
   case spv::Op::OpIMul:
   case spv::Op::OpFMul:
      return OP_MUL;
   case spv::Op::OpSDiv:
   case spv::Op::OpUDiv:
   case spv::Op::OpFDiv:
      return OP_DIV;
   case spv::Op::OpSMod:
   case spv::Op::OpUMod:
   case spv::Op::OpFMod:
      return OP_MOD;
   default:
      return OP_NOP;
   }
}

nv50_ir::CondCode
Converter::convertCc(spv::Op op)
{
   switch (op) {
   case spv::Op::OpIEqual:
   case spv::Op::OpFOrdEqual:
      return CC_EQ;
   case spv::Op::OpINotEqual:
   case spv::Op::OpFOrdNotEqual:
      return CC_NE;
   case spv::Op::OpSGreaterThan:
   case spv::Op::OpUGreaterThan:
   case spv::Op::OpFOrdGreaterThan:
      return CC_GT;
   case spv::Op::OpFUnordGreaterThan:
      return CC_GTU;
   case spv::Op::OpSGreaterThanEqual:
   case spv::Op::OpUGreaterThanEqual:
   case spv::Op::OpFOrdGreaterThanEqual:
      return CC_GE;
   case spv::Op::OpFUnordGreaterThanEqual:
      return CC_GEU;
   case spv::Op::OpSLessThan:
   case spv::Op::OpULessThan:
   case spv::Op::OpFOrdLessThan:
      return CC_LT;
   case spv::Op::OpFUnordLessThan:
      return CC_LTU;
   case spv::Op::OpSLessThanEqual:
   case spv::Op::OpULessThanEqual:
   case spv::Op::OpFOrdLessThanEqual:
      return CC_LE;
   case spv::Op::OpFUnordLessThanEqual:
      return CC_LEU;
   default:
      return CC_NO;
   }
}

bool
Converter::convertInstruction(spv::Op opcode, unsigned int numWords,
                              unsigned int firstWord)
{
   auto getOp = [&](spv::Id id, unsigned c = 0u, bool constants_allowed = true){
      auto searchOp = spvValues.find(id);
      if (searchOp == spvValues.end())
         return PValue();

      auto& opStruct = searchOp->second;
      if (c >= opStruct.value.size()) {
         _debug_printf("Trying to access element %u out of %u\n", c, opStruct.value.size());
         return PValue();
      }

      auto const pvalue = opStruct.value[c];
      auto op = pvalue.value;
      if (opStruct.storageFile == SpirvFile::IMMEDIATE) {
         if (!constants_allowed)
            return PValue();
         auto constant = op;
         op = getScratch(constant->reg.size);
         mkMov(op, constant, constant->reg.type);
         return PValue(op);
      }
      return pvalue;
   };
   auto getType = [&](spv::Id id, unsigned c = 0u){
      auto searchType = spvValues.find(id);
      if (searchType != spvValues.end()) {
         auto& opStruct = searchType->second;
         if (c < opStruct.value.size()) {
            if (opStruct.value.size() == 1)
               return opStruct.type;
            else
               return opStruct.type->getElementType(c);
         }
         _debug_printf("Trying to access element %u out of %u\n", c, opStruct.value.size());
         return static_cast<Type const*>(nullptr);
      }

      return static_cast<Type const*>(nullptr);
   };

   switch (opcode) {
   case spv::Op::OpCapability:
      if (numWords != 2u) {
         _debug_printf("OpCapability expects 2 operands but got %u\n", numWords);
         return false;
      }
      {
         using Cap = spv::Capability;
         Cap capability = getWord<Cap>(firstWord);
         if (info->target < 0xc0) {
            return true;

            if (capability == Cap::Tessellation || capability == Cap::Vector16 ||
                capability == Cap::Float16Buffer || capability == Cap::Float16 ||
                capability == Cap::Float64 || capability == Cap::Int64 ||
                capability == Cap::Int64Atomics || capability == Cap::Int16 ||
                capability == Cap::TessellationPointSize || capability == Cap::Int8 ||
                capability == Cap::TransformFeedback) {
               _debug_printf("Capability unsupported: %u\n", capability);
               return false;
            }
         }
      }
      break;
   case spv::Op::OpExtInstImport:
      if (numWords < 3u) {
         _debug_printf("OpExtInstImport expects at least 3 operands but got %u\n", numWords);
         return false;
      }
      {
         spv::Id result = getWord<spv::Id>(firstWord);
         std::string setName = binary + (firstWord + 1u) * sizeof(uint32_t);
         if (setName.empty()) {
            _debug_printf("Couldn't parse the name of OpExtInstImport at word %u\n", firstWord);
            return false;
         }
         if (setName != "OpenCL.std") {
            _debug_printf("OpExtInstImport \"%s\" is unsupported\n", setName.c_str());
            return false;
         }
         extInstructions.emplace(result, setName);
      }
      break;
   case spv::Op::OpExtInst:
      if (numWords < 5u) {
         _debug_printf("OpExtInst expects at least 5 operands but got %u\n", firstWord);
         return false;
      }
      {
         auto id = getWord<spv::Id>(firstWord + 1u);
         auto searchType = types.find(getWord<spv::Id>(firstWord));
         if (searchType == types.end()) {
            _debug_printf("Couldn't find type used by OpExInst at word %u\n", firstWord);
            return false;
         }
         auto searchExt = extInstructions.find(getWord<spv::Id>(firstWord + 2u));
         if (searchExt == extInstructions.end()) {
            _debug_printf("Couldn't find extension set used by ExtInst at word %u\n", firstWord);
            return false;
         }
         auto const op = getWord<spv::Id>(firstWord + 3u);
         if (searchExt->second == "OpenCL.std") {
            return convertOpenCLInstruction(id, searchType->second, op, firstWord, numWords);
         } else {
            _debug_printf("Unsupported extension set: \"%s\"\n", searchExt->second.c_str());
            return false;
         }
      }
      break;
   case spv::Op::OpMemoryModel:
      if (numWords != 3u) {
         _debug_printf("OpMemoryModel expects 3 operands but got %u\n", numWords);
         return false;
      }
      addressingModel = getWord<spv::AddressingModel>(firstWord);
      memoryModel = getWord<spv::MemoryModel>(firstWord + 1u);
      break;
   case spv::Op::OpEntryPoint:
      return convertEntryPoint(numWords, firstWord);
   // TODO(pmoreau): Properly handle the different execution modes
   case spv::Op::OpExecutionMode:
      if (numWords < 3u) {
         _debug_printf("OpExecutionMode expects at least 3 operands but got %u\n", numWords);
         return false;
      }
      {
         const auto entryPointId = getWord<spv::Id>(firstWord);
         const auto executionMode = getWord<spv::ExecutionMode>(firstWord + 1u);
         _debug_printf("Ignoring unsupported execution mode %u for entry point %u\n", entryPointId, executionMode);
      }
      break;
   case spv::Op::OpSource:
      return true;
   case spv::Op::OpName:
      if (numWords < 3u) {
         _debug_printf("OpName expects at least 3 operands but got %u\n", numWords);
         return false;
      }
      names.emplace(getWord<spv::Id>(firstWord), binary + (firstWord + 1u) * sizeof(uint32_t));
      break;
   case spv::Op::OpDecorate:
      return convertDecorate(numWords, firstWord);
   case spv::Op::OpMemberDecorate:
      return false;
   case spv::Op::OpDecorationGroup:
      break;
   case spv::Op::OpGroupDecorate:
      if (numWords < 2u) {
         _debug_printf("OpGroupDecorate expects at least 2 operands but got %u\n", numWords);
         return false;
      }
      {
         auto groupId = getWord<spv::Id>(firstWord);
         auto searchGroup = decorations.find(groupId);
         if (searchGroup == decorations.end()) {
            _debug_printf("DecorationGroup %u was not defined\n", groupId);
            return false;
         }

         for (unsigned int i = 1u; i < numWords; ++i) {
            auto targetId = getWord<spv::Id>(firstWord + i);
            auto &idDecorations = decorations[targetId];
            for (const auto &k : searchGroup->second)
               idDecorations[k.first].insert(idDecorations[k.first].end(), k.second.begin(), k.second.end());
         }
      }
      break;
   case spv::Op::OpTypeVoid:
      return convertType<TypeVoid>(numWords, firstWord);
   case spv::Op::OpTypeBool:
      return convertType<TypeBool>(numWords, firstWord);
   case spv::Op::OpTypeInt:
      return convertType<TypeInt>(numWords, firstWord);
   case spv::Op::OpTypeFloat:
      return convertType<TypeFloat>(numWords, firstWord);
   case spv::Op::OpTypeStruct:
      return convertType<TypeStruct>(numWords, firstWord);
   case spv::Op::OpTypeVector:
      return convertType<TypeVector>(numWords, firstWord);
   case spv::Op::OpTypeArray:
      return convertType<TypeArray>(numWords, firstWord);
   case spv::Op::OpTypePointer:
      return convertType<TypePointer>(numWords, firstWord);
   case spv::Op::OpTypeFunction:
      return convertType<TypeFunction>(numWords, firstWord);
   case spv::Op::OpTypeSampler:
      return convertType<TypeSampler>(numWords, firstWord);
   case spv::Op::OpTypeImage:
      return convertType<TypeImage>(numWords, firstWord);
   case spv::Op::OpTypeSampledImage:
      return convertType<TypeSampledImage>(numWords, firstWord);
   case spv::Op::OpConstant:
      if (numWords < 3u) {
         _debug_printf("OpConstant expects at least 3 operands but got %u\n", numWords);
         return false;
      }
      {
         auto id = getWord<spv::Id>(firstWord + 1u);
         auto search = types.find(getWord<spv::Id>(firstWord));
         if (search == types.end()) {
            _debug_printf("Couldn't find type used by OpConstant at word %u\n", firstWord);
            return false;
         }
         Words values = Words();
         for (unsigned int i = 2u; i < numWords - 1u; ++i)
            values.push_back(getWord<unsigned>(firstWord + i));
         unsigned int position = 0u;
         auto constants = search->second->generateConstant(values, *this, position);
         std::vector<PValue> res;
         for (auto c : constants)
            res.push_back(c);
         spvValues.emplace(id, SpirVValue{ SpirvFile::IMMEDIATE, search->second, res, search->second->getPaddings() });
      }
      break;
   case spv::Op::OpConstantNull:
      {
         auto id = getWord<spv::Id>(firstWord + 1u);
         auto search = types.find(getWord<spv::Id>(firstWord));
         if (search == types.end()) {
            _debug_printf("Couldn't find type used by OpConstant at word %u\n", firstWord);
            return false;
         }
         auto constants = search->second->generateNullConstant(*this);
         std::vector<PValue> res;
         for (auto c : constants)
            res.push_back(c);
         spvValues.emplace(id, SpirVValue{ SpirvFile::IMMEDIATE, search->second, res, search->second->getPaddings() });
      }
   case spv::Op::OpConstantComposite:
      if (numWords < 3u) {
         _debug_printf("OpConstantComposite expects at least 3 operands but got %u\n", numWords);
         return false;
      }
      {
         auto id = getWord<spv::Id>(firstWord + 1u);
         auto search = types.find(getWord<spv::Id>(firstWord));
         if (search == types.end()) {
            _debug_printf("Couldn't find type used by OpConstant at word %u\n", firstWord);
            return false;
         }
         auto values = std::vector<PValue>();
         for (unsigned int i = 2u; i < numWords - 1u; ++i) {
            auto elemId = getWord<spv::Id>(firstWord + i);
            auto searchElem = spvValues.find(elemId);
            if (searchElem == spvValues.end()) {
               _debug_printf("Couldn't find constant %u for constant composite %u\n", elemId, id);
               return false;
            }
            values.insert(values.end(), searchElem->second.value.cbegin(), searchElem->second.value.cend());
         }
         spvValues.emplace(id, SpirVValue{ SpirvFile::IMMEDIATE, search->second, values, search->second->getPaddings() });
      }
      break;
   case spv::Op::OpConstantSampler:
      {
         if (numWords != 6u) {
            _debug_printf("OpConstantSampler expects at 6 operands but got %u\n", numWords);
            return false;
         }

         auto const typeId = getWord<spv::Id>(firstWord);
         auto const searchType = types.find(typeId);
         if (searchType == types.end()) {
            _debug_printf("Couldn't find type %u used by OpConstantSampler at word %u\n", typeId, firstWord);
            return false;
         }
         auto const resId = getWord<spv::Id>(firstWord + 1u);
         auto const addressingMode = getWord<spv::SamplerAddressingMode>(firstWord + 2u);
         auto const param = getWord<unsigned>(firstWord + 3u);
         auto const filterMode = getWord<spv::SamplerFilterMode>(firstWord + 4u);

         auto const usesNormalizedCoords = param == 0u;

         samplers.emplace(resId, Sampler{ reinterpret_cast<TypeSampler const*>(searchType->second), addressingMode, usesNormalizedCoords, filterMode });
      }
      break;
   case spv::Op::OpVariable:
      if (numWords < 4u) {
         _debug_printf("OpVariable expects at least 4 operands but got %u\n", numWords);
         return false;
      }
      {
         auto search = types.find(getWord<spv::Id>(firstWord));
         if (search == types.end()) {
            _debug_printf("Couldn't find type used by OpVariable at word %u\n", firstWord);
            return false;
         }
         auto id = getWord<spv::Id>(firstWord + 1u);
         auto storage_file = getStorageFile(getWord<spv::StorageClass>(firstWord + 2u));

         auto ptr = static_cast<TypePointer const*>(search->second);
         auto isBuiltIn = false;
         auto search_decorations = decorations.find(id);
         if (search_decorations != decorations.end()) {
            isBuiltIn = search_decorations->second.find(spv::Decoration::BuiltIn) != search_decorations->second.end();
            auto search_linkage = search_decorations->second.find(spv::Decoration::BuiltIn);
            if (!isBuiltIn && search_linkage != search_decorations->second.end() && static_cast<spv::LinkageType>(search_linkage->second[0][0]) == spv::LinkageType::Import) {
               _debug_printf("Variable %u has linkage type \"import\"! Missing a link step?\n", id);
               return false;
            }
         }

         if (numWords == 5u) {
            auto init_id = getWord<spv::Id>(firstWord + 3u);
            auto searchObject = spvValues.find(init_id);
            if (searchObject == spvValues.end()) {
               _debug_printf("Couldn't find initial value %u for variable %u\n", init_id, id);
               return false;
            }

            // If we have an immediate, which are stored in const memory,
            // inline it
            if (storage_file == SpirvFile::CONST && searchObject->second.storageFile == SpirvFile::IMMEDIATE)
               spvValues.emplace(id, SpirVValue{ SpirvFile::IMMEDIATE, ptr, searchObject->second.value, searchObject->second.paddings });
            else
               spvValues.emplace(id, SpirVValue{ storage_file, ptr, searchObject->second.value, searchObject->second.paddings });
         } else if (!isBuiltIn) {
            acquire(storage_file, id, ptr);
         }
      }
      break;
   case spv::Op::OpNop:
      break;
   case spv::Op::OpUndef:
      if (numWords != 3u) {
         _debug_printf("OpUndef expects 3 operands but got %u\n", numWords);
         return false;
      }
      {
         auto search = types.find(getWord<spv::Id>(firstWord));
         if (search == types.end()) {
            _debug_printf("Couldn't find type used by OpUndef at word %u\n", firstWord);
            return false;
         }
         auto id = getWord<spv::Id>(firstWord + 1u);

         auto constants = search->second->generateNullConstant(*this);
         std::vector<PValue> res;
         for (auto i : constants)
            res.push_back(i);
         spvValues.emplace(id, SpirVValue{ SpirvFile::IMMEDIATE, search->second, res, search->second->getPaddings() });
      }
      break;
   // TODO:
   // * use FunctionControl
   // * use decorations
   case spv::Op::OpFunction:
      if (numWords != 5u) {
         _debug_printf("OpFunction expects 5 operands but got %u\n", numWords);
         return false;
      }
      if (func != nullptr) {
         _debug_printf("Defining a function inside another function is not allowed!\n");
         return false;
      }
      {
         spv::Id id = getWord<spv::Id>(firstWord + 1u);
         auto search_func = functions.find(id);
         if (search_func != functions.end()) {
            func = search_func->second;
            setPosition(BasicBlock::get(func->cfg.getRoot()), true);
            return true;
         }

         auto resTypeId = getWord<spv::Id>(firstWord);
         auto resTypeIter = types.find(resTypeId);
         if (resTypeIter == types.end()) {
            _debug_printf("Couldn't find return type %u of function %u\n", resTypeId, id);
            return false;
         }

         using FCM = spv::FunctionControlMask;
         FCM control = getWord<FCM>(firstWord + 2u);

         auto search_name = names.find(id);
         if (search_name == names.end()) {
            _debug_printf("Couldn't find a name for function at word %u\n", firstWord);
            return false;
         }
         auto search_entry = entryPoints.find(id);
         auto const label = (search_entry == entryPoints.end()) ? UINT32_MAX : search_entry->second.index; // XXX valid symbol needed for functions?
         auto &name = search_name->second;
         char *func_name = new char[name.size() + 1u];
         std::strncpy(func_name, name.c_str(), name.size() + 1u);
         Function *function = new Function(prog, func_name, label);
         functions.emplace(id, function);
         func = function;
         currentFuncId = id;

         prog->main->call.attach(&func->call, Graph::Edge::TREE); // XXX only bind entry points to main?
         BasicBlock *block = new BasicBlock(func);
         func->setEntry(block);
         func->setExit(new BasicBlock(func));
         blocks.emplace(static_cast<spv::Id>(0u), block);
         prog->calls.insert(&func->call);

         if (!resTypeIter->second->isVoidType())
            func->outs.emplace_back(getScratch(resTypeIter->second->getSize()));

         setPosition(block, true);
      }
      break;
   // TODO:
   // * use decorations
   case spv::Op::OpFunctionParameter:
      if (numWords != 3u) {
         _debug_printf("OpFunctionParameter expects 3 operands but got %u\n", numWords);
         return false;
      }
      if (func == nullptr) {
         _debug_printf("Defining function parameters outside of function definition is not allowed\n");
         return false;
      }
      {
         spv::Id id = getWord<spv::Id>(firstWord + 1u);
         spv::Id type = getWord<spv::Id>(firstWord);
         auto search = types.find(type);
         if (search == types.end()) {
            _debug_printf("Couldn't find type associated to id %u\n", type);
            return false;
         }
         bool isKernel = false;
         auto search_entry = entryPoints.find(currentFuncId);
         if (search_entry != entryPoints.end())
            isKernel = (search_entry->second.executionModel == spv::ExecutionModel::Kernel);

         Type *paramType = search->second;
         SpirvFile destStorageFile = (paramType->getType() != spv::Op::OpTypePointer) ? SpirvFile::TEMPORARY : reinterpret_cast<const TypePointer *>(paramType)->getStorageFile();
         auto decos = decorations.find(id);
         if (decos != decorations.end()) {
            auto paramAttrs = decos->second.find(spv::Decoration::FuncParamAttr);
            if (paramAttrs != decos->second.end() && static_cast<spv::FunctionParameterAttribute>(paramAttrs->second[0][0]) == spv::FunctionParameterAttribute::ByVal) {
               paramType = reinterpret_cast<const TypePointer *>(search->second)->getPointedType();
               destStorageFile = SpirvFile::TEMPORARY;
            }
         }
         if (isKernel) {
            inputOffset += load(destStorageFile, SpirvFile::INPUT, id, PValue(nullptr, nullptr), inputOffset, paramType);
            spvValues[id].type = search->second;
         } else {
            std::vector<PValue> values;
            std::stack<Type const*> stack;
            stack.push(paramType);
            while (!stack.empty()) {
               unsigned deltaOffset = 0u;
               auto currentType = stack.top();
               stack.pop();
               if (!currentType->isCompooundType()) {
                  Value *res = getScratch(std::max(4u, currentType->getSize()));
                  values.emplace_back(nullptr, res);
                  func->ins.emplace_back(res);
               } else {
                  for (unsigned int i = currentType->getElementsNb(); i != 0u; --i)
                     stack.push(currentType->getElementType(i - 1u));
               }
            }
            spvValues.emplace(id, SpirVValue{ SpirvFile::TEMPORARY, paramType, values, paramType->getPaddings() });
         }
      }
      break;
   case spv::Op::OpFunctionEnd:
      if (numWords != 1u) {
         _debug_printf("OpFunctionEnd expects 1 operand but got %u\n", numWords);
         return false;
      }
      if (func == nullptr) {
         _debug_printf("Reached end of function while not defining one\n");
         return false;
      }
      {
         for (auto i : phiToMatch) {
            auto phiId = i.first;
            auto searchPhiData = phiNodes.find(phiId);
            if (searchPhiData == phiNodes.end()) {
               _debug_printf("Couldn't find phi data for id %u\n", phiId);
               return false;
            }
            for (auto j : i.second) {
               auto index = j.first;
               auto varId = j.second.first;
               auto& varBbPair = searchPhiData->second[index];
               if (varId != 0u) {
                  auto searchVar = spvValues.find(varId);
                  if (searchVar == spvValues.end()) {
                     _debug_printf("Couldn't find variable with id %u\n", varId);
                     return false;
                  }
                  varBbPair.first = searchVar->second.value;
                  _debug_printf("Found var with id %u: %p\n", varId, varBbPair.first[0]);
               }
               auto bbId = j.second.second;
               if (bbId != 0u) {
                  auto searchBb = blocks.find(bbId);
                  if (searchBb == blocks.end()) {
                     _debug_printf("Couldn't find BB with id %u\n", bbId);
                     return false;
                  }
                  varBbPair.second = searchBb->second;
                  _debug_printf("Found bb with id %u: %p\n", bbId, varBbPair.second);
               }
            }
         }
         phiToMatch.clear();

         // Debugging purposes
         {
            for (auto block : blocks) {
               auto lBb = block.second;
               Instruction *next;
               for (Instruction *i = lBb->getPhi(); i && i != lBb->getEntry(); i = next) {
                  next = i->next;
                  auto searchPhi = phiMapping.find(i);
                  if (searchPhi == phiMapping.end()) {
                     assert(false);
                     return false;
                  }
                  auto searchPhiData = phiNodes.find(searchPhi->second);
                  if (searchPhiData == phiNodes.end()) {
                     assert(false);
                     return false;
                  }
                  auto counter = 0u;
                  for (auto& phiPair : searchPhiData->second) {
                     i->setSrc(0, phiPair.first[0].value);
                     ++counter;
                  }
               }
            }
         }

         if (!branchesToMatch.empty()) {
            _debug_printf("Could not match some branches!\n");
            for (auto const& i : branchesToMatch) {
               _debug_printf("\t%u: ", i.first);
               for (auto const& j : i.second)
                  _debug_printf("%p ", j);
               _debug_printf("\n");
            }
         }

         BasicBlock *leave = BasicBlock::get(func->cfgExit);
         setPosition(leave, true);
         if (entryPoints.find(static_cast<spv::Id>(func->getLabel())) != entryPoints.end())
            mkFlow(OP_EXIT, nullptr, CC_ALWAYS, nullptr)->fixed = 1;
         else
            mkFlow(OP_RET, nullptr, CC_ALWAYS, nullptr)->fixed = 1;

         blocks.clear();
         func = nullptr;
         currentFuncId = 0u;
         inputOffset = 0u;
      }
      break;
   case spv::Op::OpFunctionCall:
      if (numWords < 4u) {
         _debug_printf("OpFunctionCall expects at lest 4 operands but got %u\n", numWords);
         return false;
      }
      {
         auto resTypeId = getWord<spv::Id>(firstWord);
         auto resTypeIter = types.find(resTypeId);
         if (resTypeIter == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", resTypeId);
            return false;
         }
         auto resType = resTypeIter->second;

         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto functionId = getWord<spv::Id>(firstWord + 2u);

         auto insn = mkFlow(OP_CALL, nullptr, CC_ALWAYS, NULL);

         for (size_t i = 3u, j = 0u; i < numWords - 1u; ++i, ++j) {
            auto const argId = getWord<spv::Id>(firstWord + i);
            auto argIter = spvValues.find(argId);
            if (argIter == spvValues.end()) {
               _debug_printf("Couldn't not find %uth argument %u of function call %u\n", j, argId, functionId);
               return false;
            }
            insn->setSrc(i - 3u, argIter->second.value.front().value);
         }

         Instruction *resInsn = nullptr;
         Value *res = nullptr;
         if (!resType->isVoidType()) {
            res = getScratch(resType->getSize());
            insn->setDef(0, res);
            spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, resType, { res }, resType->getPaddings() });
         }

         auto functionToMatchIter = functionsToMatch.find(functionId);
         if (functionToMatchIter == functionsToMatch.end())
            functionsToMatch.emplace(functionId, std::vector<FunctionData>{ FunctionData{ func, insn } });
         else
            functionToMatchIter->second.emplace_back(func, insn);
      }
      break;
   case spv::Op::OpLabel:
      if (numWords != 2u) {
         _debug_printf("OpLabel expects 2 operands but got %u\n", numWords);
         return false;
      }
      if (bb != nullptr && blocks.size() != 1u) {
         _debug_printf("Defining a block inside another block is not allowed\n");
         return false;
      }
      {
         BasicBlock *block = new BasicBlock(func);

         // Attach to first block, which is the block containing the loading of
         // the function parameters.
         if (blocks.size() == 1u) {
            mkFlow(OP_BRA, block, CC_ALWAYS, nullptr);
            blocks[0u]->cfg.attach(&block->cfg, Graph::Edge::TREE);
         }

         auto id = getWord<spv::Id>(firstWord);
         auto searchFlows = branchesToMatch.find(id);
         if (searchFlows != branchesToMatch.end()) {
            for (auto& flow : searchFlows->second) {
               flow->bb->getExit()->asFlow()->target.bb = block;
               flow->bb->cfg.attach(&block->cfg, (block->cfg.incidentCount() == 0u) ? Graph::Edge::TREE : Graph::Edge::FORWARD);
            }
            branchesToMatch.erase(searchFlows);
         }

         blocks.emplace(id, block);
         setPosition(block, true);
      }
      break;
   case spv::Op::OpReturn:
      if (numWords != 1u) {
         _debug_printf("OpReturn expects 1 operand but got %u\n", numWords);
         return false;
      }
      if (bb == nullptr) {
         _debug_printf("Reached end of block while not defining one\n");
         return false;
      }
      {
         BasicBlock *leave = BasicBlock::get(func->cfgExit);
         mkFlow(OP_BRA, leave, CC_ALWAYS, nullptr);
         bb->cfg.attach(&leave->cfg, (leave->cfg.incidentCount() == 0) ? Graph::Edge::TREE : Graph::Edge::FORWARD);

         bb = nullptr;
      }
      break;
   case spv::Op::OpReturnValue:
      if (numWords != 2u) {
         _debug_printf("OpReturnValue expects 2 operand but got %u\n", numWords);
         return false;
      }
      if (bb == nullptr) {
         _debug_printf("Reached end of block while not defining one\n");
         return false;
      }
      {
         auto retId = getWord<spv::Id>(firstWord);
         auto retIter = spvValues.find(retId);
         if (retIter == spvValues.end()) {
            _debug_printf("Couldn't find value %u returned at word %u\n", retId, firstWord);
            return false;
         }
         assert(func->outs.size() == 1u);
         mkOp1(OP_MOV, retIter->second.type->getEnumType(), func->outs.front().get(), retIter->second.value.front().value);

         BasicBlock *leave = BasicBlock::get(func->cfgExit);
         mkFlow(OP_BRA, leave, CC_ALWAYS, nullptr);
         bb->cfg.attach(&leave->cfg, (leave->cfg.incidentCount() == 0) ? Graph::Edge::TREE : Graph::Edge::FORWARD);

         bb = nullptr;
      }
      break;
   case spv::Op::OpBranch:
      if (numWords != 2u) {
         _debug_printf("OpBranch expects 2 operands but got %u\n", numWords);
         return false;
      }
      if (bb == nullptr) {
         _debug_printf("Reached end of block while not defining one\n");
         return false;
      }
      {
         auto labelId = getWord<spv::Id>(firstWord);
         auto searchLabel = blocks.find(labelId);
         if (searchLabel == blocks.end()) {
            auto flow = mkFlow(OP_BRA, nullptr, CC_ALWAYS, nullptr);
            auto searchFlow = branchesToMatch.find(labelId);
            if (searchFlow == branchesToMatch.end())
               branchesToMatch.emplace(labelId, std::vector<FlowInstruction*>{ flow });
            else
               searchFlow->second.push_back(flow);
         } else {
            mkFlow(OP_BRA, searchLabel->second, CC_ALWAYS, nullptr);
            bb->cfg.attach(&searchLabel->second->cfg, Graph::Edge::BACK);
         }

         bb = nullptr;
      }
      break;
   case spv::Op::OpBranchConditional:
      if (numWords < 4u) {
         _debug_printf("OpBranchConditional expects at least 4 operands but got %u\n", numWords);
         return false;
      }
      if (bb == nullptr) {
         _debug_printf("Reached end of block while not defining one\n");
         return false;
      }
      {
         auto predId = getWord<spv::Id>(firstWord);
         auto ifId = getWord<spv::Id>(firstWord + 1u);
         auto elseId = getWord<spv::Id>(firstWord + 2u);

         auto searchPred = spvValues.find(predId);
         if (searchPred == spvValues.end()) {
            _debug_printf("Couldn't find predicate with id %u\n", predId);
            return false;
         }
         auto pred = searchPred->second.value[0].value;
         auto searchIf = blocks.find(ifId);
         if (searchIf == blocks.end()) {
            auto flow = mkFlow(OP_BRA, nullptr, CC_P, pred);
            auto searchFlow = branchesToMatch.find(ifId);
            if (searchFlow == branchesToMatch.end())
               branchesToMatch.emplace(ifId, std::vector<FlowInstruction*>{ flow });
            else
               searchFlow->second.push_back(flow);
         } else {
            mkFlow(OP_BRA, searchIf->second, CC_P, pred);
            bb->cfg.attach(&searchIf->second->cfg, Graph::Edge::BACK);
         }

         auto tmp = new BasicBlock(func);
         bb->cfg.attach(&tmp->cfg, Graph::Edge::TREE);
         setPosition(tmp, true);

         auto searchElse = blocks.find(elseId);
         if (searchElse == blocks.end()) {
            auto flow = mkFlow(OP_BRA, nullptr, CC_ALWAYS, nullptr);
            auto searchFlow = branchesToMatch.find(elseId);
            if (searchFlow == branchesToMatch.end())
               branchesToMatch.emplace(elseId, std::vector<FlowInstruction*>{ flow });
            else
               searchFlow->second.push_back(flow);
         } else {
            mkFlow(OP_BRA, searchElse->second, CC_ALWAYS, nullptr);
            bb->cfg.attach(&searchElse->second->cfg, Graph::Edge::BACK);
         }

         bb = nullptr;
      }
      break;
   case spv::Op::OpPhi:
      if (numWords <= 3u) {
         _debug_printf("OpPhi expects at least 3 operands but got %u\n", numWords);
         return false;
      }
      {
         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto searchType = types.find(typeId);
         if (searchType == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }
         auto type = searchType->second;
         auto parents = std::vector<std::pair<std::vector<PValue>, BasicBlock*>>();
         auto toMatchs = std::unordered_map<uint32_t, std::pair<spv::Id, spv::Id>>();
         for (unsigned int i = 2u, counter = 0u; i < numWords - 2u; i += 2u, ++counter) {
            auto vars = std::vector<PValue>();
            auto varId = getWord<spv::Id>(firstWord + i);
            auto toMatch = std::make_pair<spv::Id, spv::Id>(0u, 0u);
            for (unsigned int j = 0u; j < type->getElementsNb(); ++j) {
               auto var = getOp(varId, j, false).value;
               if (var == nullptr) {
                  _debug_printf("Couldn't find variable with id %u, keeping looking for it\n", varId);
                  toMatch.first = varId;
               }
               vars.push_back(var);
            }
            auto bbId = getWord<spv::Id>(firstWord + i + 1);
            auto searchBB = blocks.find(bbId);
            if (searchBB == blocks.end()) {
               _debug_printf("Couldn't find BB with id %u, keeping looking for it\n", bbId);
               toMatch.second = bbId;
            }
            if (toMatch.first != 0u || toMatch.second != 0u)
               toMatchs.emplace(counter, toMatch);
            parents.emplace_back(vars, (searchBB != blocks.end()) ? searchBB->second : nullptr);
         }
         auto value = std::vector<PValue>();
         if (type->getElementsNb() > 1u)
            _debug_printf("OpPhi on type with more than 1 element: need to check behaviour!\n");
         for (unsigned int i = 0u; i < type->getElementsNb(); ++i)
            value.push_back(getScratch(type->getElementSize(i)));
         auto phi = new_Instruction(func, OP_PHI, nv50_ir::TYPE_U32); // This instruction will be removed later, so we don't care about the size
         phi->setDef(0, value[0].value);
         bb->insertTail(phi);
         phiNodes.emplace(resId, parents);
         phiMapping.emplace(phi, resId);
         if (!toMatchs.empty())
            phiToMatch.emplace(resId, toMatchs);
         spvValues.emplace(resId, SpirVValue{ SpirvFile::NONE, type, value, type->getPaddings() });
      }
      break;
   case spv::Op::OpSwitch:
      if (numWords < 3u) {
         _debug_printf("OpSwitch expects at least 3 operands but got %u\n", numWords);
         return false;
      }
      {
         spv::Id selectorId = getWord<spv::Id>(firstWord);
         auto search_selector = spvValues.find(selectorId);
         if (search_selector == spvValues.end()) {
            _debug_printf("Could not find selector with id %u\n", selectorId);
            return false;
         }
         auto type = search_selector->second.type;
         auto const width = type->getSize() / 4u; // FIXME need to round it to upper
         BasicBlock *new_bb = bb;
         BasicBlock *old_bb = bb;
         for (size_t i = 2u; i < numWords - 1u; i += width + 1u) {
            Words values = Words();
            for (unsigned int j = 0u; j < width; ++j)
               values.push_back(getWord<unsigned>(firstWord + i + j));
            unsigned int position = 0u;
            auto imm = type->generateConstant(values, *this, position).front();
            auto imm2 = getScratch(type->getSize());
            mkMov(imm2, imm, type->getEnumType());
            auto const label_id = getWord<spv::Id>(firstWord + i + width);
            auto pred = getScratch(1, FILE_PREDICATE);
            mkCmp(OP_SET, CC_EQ, TYPE_U32, pred, type->getEnumType(), search_selector->second.value[0].value, imm2);
            auto search_label = blocks.find(label_id);
            if (search_label == blocks.end()) {
               auto flow = mkFlow(OP_BRA, nullptr, CC_P, pred);
               auto searchFlow = branchesToMatch.find(label_id);
               if (searchFlow == branchesToMatch.end())
                  branchesToMatch.emplace(label_id, std::vector<FlowInstruction*>{ flow });
               else
                  searchFlow->second.push_back(flow);
            } else {
               mkFlow(OP_BRA, search_label->second, CC_P, pred);
               old_bb->cfg.attach(&search_label->second->cfg, Graph::Edge::BACK);
            }
            new_bb = new BasicBlock(func);
            old_bb->cfg.attach(&new_bb->cfg, Graph::Edge::TREE);
            setPosition(new_bb, true);
            old_bb = new_bb;
         }

         auto const default_id = getWord<spv::Id>(firstWord + 1u);
         auto search_label = blocks.find(default_id);
         if (search_label == blocks.end()) {
            auto flow = mkFlow(OP_BRA, nullptr, CC_ALWAYS, nullptr);
            auto searchFlow = branchesToMatch.find(default_id);
            if (searchFlow == branchesToMatch.end())
               branchesToMatch.emplace(default_id, std::vector<FlowInstruction*>{ flow });
            else
               searchFlow->second.push_back(flow);
         } else {
            mkFlow(OP_BRA, search_label->second, CC_ALWAYS, nullptr);
            new_bb->cfg.attach(&search_label->second->cfg, Graph::Edge::BACK);
         }

         bb = nullptr;
      }
      break;
   case spv::Op::OpLoad:
      if (numWords < 4u) {
         _debug_printf("OpLoad expects at least 4 operands but got %u\n", numWords);
         return false;
      }
      {
         auto type_id = getWord<spv::Id>(firstWord);
         auto searchType = types.find(type_id);
         if (searchType == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", type_id);
            return false;
         }
         auto type = searchType->second;
         spv::Id resId = getWord<spv::Id>(firstWord + 1u);
         spv::Id pointerId = getWord<spv::Id>(firstWord + 2u);
         spv::MemoryAccessMask access = spv::MemoryAccessMask::MaskNone;
         if (numWords == 5u)
            access = getWord<spv::MemoryAccessMask>(firstWord + 3u);
         uint32_t alignment = 0u;
         if (hasFlag(access, spv::MemoryAccessShift::Aligned)) {
            if (numWords != 6u) {
               _debug_printf("OpLoad with an aligned memory access flag expects 6 operands but got %u\n", numWords);
               return false;
            }
            alignment = getWord<unsigned>(firstWord + 4u);
         }

         auto search_decorations = decorations.find(pointerId);
         if (search_decorations != decorations.end()) {
            for (auto& decoration : search_decorations->second) {
               if (decoration.first != spv::Decoration::BuiltIn)
                  continue;

               return loadBuiltin(resId, type, decoration.second.front(), access);
            }
         }

         auto search_pointer = spvValues.find(pointerId);
         if (search_pointer == spvValues.end()) {
            _debug_printf("Couldn't find pointer with id %u\n", pointerId);
            return false;
         }

         auto pointer_type = reinterpret_cast<const TypePointer *>(search_pointer->second.type);
         load(SpirvFile::TEMPORARY, pointer_type->getStorageFile(), resId, search_pointer->second.value[0], 0u, type, access, alignment);
      }
      break;
   case spv::Op::OpStore:
      if (numWords < 3u) {
         _debug_printf("OpStore expects at least 3 operands but got %u\n", numWords);
         return false;
      }
      {
         spv::Id pointerId = getWord<spv::Id>(firstWord);
         spv::Id objectId = getWord<spv::Id>(firstWord + 1u);
         spv::MemoryAccessMask access = spv::MemoryAccessMask::MaskNone;
         if (numWords == 4u)
            access = getWord<spv::MemoryAccessMask>(firstWord + 2u);
         uint32_t alignment = 0u;
         if (hasFlag(access, spv::MemoryAccessShift::Aligned)) {
            if (numWords != 5u) {
               _debug_printf("OpStore with an aligned memory access flag expects 5 operands but got %u\n", numWords);
               return false;
            }
            alignment = getWord<unsigned>(firstWord + 3u);
         }

         auto search_pointer = spvValues.find(pointerId);
         if (search_pointer == spvValues.end()) {
            _debug_printf("Couldn't find pointer with id %u\n", pointerId);
            return false;
         }

         auto pointer_type = reinterpret_cast<const TypePointer *>(search_pointer->second.type);
         auto searchElementStruct = spvValues.find(objectId);
         if (searchElementStruct == spvValues.end()) {
            _debug_printf("Couldn't find object with id %u\n", objectId);
            return false;
         }

         auto value = searchElementStruct->second.value;
         store(pointer_type->getStorageFile(), search_pointer->second.value[0], 0u, value, pointer_type->getPointedType(), access, alignment);
      }
      break;
   case spv::Op::OpPtrAccessChain: // FALLTHROUGH
   case spv::Op::OpInBoundsPtrAccessChain:
      if (numWords < 5u) {
         _debug_printf("OpInBoundsPtrAccessChain expects at least 5 operands but got %u\n", numWords);
         return false;
      }
      {
         auto resTypeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto baseId = getWord<spv::Id>(firstWord + 2u);

         auto searchBaseStruct = spvValues.find(baseId);
         if (searchBaseStruct == spvValues.end()) {
            _debug_printf("Couldn't find base with id %u\n", baseId);
            return false;
         }
         auto& baseStruct = searchBaseStruct->second;
         auto base = baseStruct.value[0];
         auto baseType = baseStruct.type;

         std::vector<Value *> indices;
         for (unsigned int i = 3u; i < numWords - 1u; ++i) {
            auto elementId = getWord<spv::Id>(firstWord + i);
            auto searchElementStruct = spvValues.find(elementId);
            if (searchElementStruct == spvValues.end()) {
               _debug_printf("Couldn't find element with id %u\n", elementId);
               return false;
            }

            auto& elementStruct = searchElementStruct->second;
            auto element = elementStruct.value[0];
            indices.push_back(element.value);
         }

         auto resTypeIter = types.find(resTypeId);
         if (resTypeIter == types.end()) {
            _debug_printf("Couldn't find pointer type of id %u\n", resTypeId);
            return false;
         }
         auto resType = reinterpret_cast<const TypePointer *>(resTypeIter->second);

         // If in GPRs, copy from indexed element up to the last element
         // Otherwise, compute offset as before

         std::vector<PValue> values;
         std::vector<unsigned int> paddings;
         if (baseStruct.storageFile == SpirvFile::TEMPORARY || baseStruct.storageFile == SpirvFile::IMMEDIATE) {
            unsigned index = 0u, depth = 0u;
            std::stack<const Type *> typesStack;
            typesStack.push(baseStruct.type);
            while (!typesStack.empty()) {
               const Type *currentType = typesStack.top();
               typesStack.pop();
               assert(currentType->isCompooundType() || currentType->getType() == spv::Op::OpTypePointer);
               if (currentType->getType() == spv::Op::OpTypePointer) {
               } else {
                  std::function<unsigned int (const Type *type)> const &getMembersCount = [&getMembersCount](const Type *type) {
                     if (!type->isCompooundType())
                        return 1u;
                     unsigned members_count = 0u;
                     for (unsigned int i = 0u; i < type->getElementsNb(); ++i)
                        members_count += getMembersCount(type->getElementType(i));
                     return members_count;
                  };
                  int member_index = indices[depth]->reg.data.s32;
                  for (int i = member_index - 1; i >= 0; --i)
                     index += getMembersCount(currentType->getElementType(i));
                  typesStack.push(currentType->getElementType(member_index));
                  ++depth;
               }
            }
            values.insert(values.end(), baseStruct.value.begin() + index, baseStruct.value.end());
            paddings.insert(paddings.end(), baseStruct.paddings.begin() + index, baseStruct.paddings.end());
         } else {
            auto offset = getScratch((info->target < 0xc0) ? 4u : 8u);
            if (info->target < 0xc0) {
               offset->reg.type = TYPE_U32;
               loadImm(offset, 0u);
            } else {
               offset->reg.type = TYPE_U64;
               loadImm(offset, 0lu);
            }
            auto search_decorations = decorations.find(baseId);
            baseType->getGlobalOffset(this, (search_decorations != decorations.end()) ? search_decorations->second : Decoration(), offset, indices);
            if (base.isValue()) {
               auto ptr = getScratch(offset->reg.size);
               mkOp2(OP_ADD, offset->reg.type, ptr, base.value, offset);
               values.emplace_back(nullptr, ptr);
            } else {
               values.emplace_back(base.symbol, offset);
            }
            paddings.push_back(1u);
         }
         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, resType, values, paddings });
      }
      break;
   case spv::Op::OpCompositeExtract:
      if (numWords < 4u) {
         _debug_printf("OpCompositeExtract expects at least 4 operands but got %u\n", numWords);
         return false;
      }
      {
         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto baseId = getWord<spv::Id>(firstWord + 2u);

         auto searchBaseStruct = spvValues.find(baseId);
         if (searchBaseStruct == spvValues.end()) {
            _debug_printf("Couldn't find base with id %u\n", baseId);
            return false;
         }
         auto& baseStruct = searchBaseStruct->second;
         auto base = baseStruct.value[0];

         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }
         auto baseType = baseStruct.type;

         auto ids = std::vector<unsigned int>();
         for (unsigned int i = 3u; i < numWords - 1u; ++i)
            ids.push_back(getWord<unsigned int>(firstWord + i));
         auto offset = baseType->getGlobalIdx(ids);

         Value* dst = nullptr;
         if (base.isValue()) {
            auto searchSrc = spvValues.find(baseId);
            if (searchSrc == spvValues.end()) {
               _debug_printf("Member %u of id %u couldn't be found\n", offset, baseId);
               return false;
            }
            auto& value = searchSrc->second.value;
            if (offset >= value.size()) {
               _debug_printf("Trying to access member %u out of %u\n", offset, value.size());
               return false;
            }
            auto src = value[offset].value;
            dst = getScratch(type->second->getSize());
            mkMov(dst, src, type->second->getEnumType());
            spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type->second, { dst }, type->second->getPaddings() });
         } else {
            load(SpirvFile::TEMPORARY, baseStruct.storageFile, baseId, PValue(), offset, baseType);
         }
      }
      break;
   case spv::Op::OpCompositeInsert:
      if (numWords < 5u) {
         _debug_printf("OpCompositeInsert expects at least 5 operands but got %u\n", numWords);
         return false;
      }
      {
         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto objId = getWord<spv::Id>(firstWord + 2u);
         auto baseId = getWord<spv::Id>(firstWord + 3u);

         auto obj = getOp(objId);
         if (obj.isUndefined()) {
            _debug_printf("Couldn't find obj with id %u\n", objId);
            return false;
         }

         auto searchBaseStruct = spvValues.find(baseId);
         if (searchBaseStruct == spvValues.end()) {
            _debug_printf("Couldn't find base with id %u\n", baseId);
            return false;
         }
         auto& baseStruct = searchBaseStruct->second;
         auto base = baseStruct.value[0];

         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }
         auto baseType = baseStruct.type;

         auto ids = std::vector<unsigned int>();
         for (unsigned int i = 4u; i < numWords - 1u; ++i)
            ids.push_back(getWord<unsigned int>(firstWord + i));
         auto offset = baseType->getGlobalIdx(ids);

         if (base.isValue()) {
            auto searchSrc = spvValues.find(baseId);
            if (searchSrc == spvValues.end()) {
               _debug_printf("Member %u of id %u couldn't be found\n", offset, baseId);
               return false;
            }
            auto& value = searchSrc->second.value;
            if (offset >= value.size()) {
               _debug_printf("Trying to access member %u out of %u\n", offset, value.size());
               return false;
            }
            auto res = std::vector<PValue>(value.size());
            for (unsigned int i = 0u; i < value.size(); ++i) {
               res[i] = getScratch(type->second->getElementSize(i));
               if (i != offset)
                  mkMov(res[i].value, value[i].value, type->second->getElementEnumType(i));
               else
                  mkMov(res[i].value, obj.value, type->second->getElementEnumType(i));
            }
            spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type->second, res, type->second->getPaddings() });
         } else {
            _debug_printf("OpCompositeInsert is not supported yet on non-reg stored values\n");
            return false;
         }
      }
      break;
   case spv::Op::OpBitcast:
      if (numWords != 4u) {
         _debug_printf("OpBitcast expects 4 operands but got %u\n", numWords);
         return false;
      }
      {
         const auto resTypeId = getWord<spv::Id>(firstWord);
         const auto resId = getWord<spv::Id>(firstWord + 1u);
         const auto operandId = getWord<spv::Id>(firstWord + 2u);

         auto type = types.find(resTypeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", resTypeId);
            return false;
         }

         auto op = spvValues.find(operandId);
         if (op == spvValues.end()) {
            _debug_printf("Couldn't find op with id %u\n", operandId);
            return false;
         }

         auto storageFile = SpirvFile::TEMPORARY;
         const Type *resType = type->second;
         if (type->second->getType() == spv::Op::OpTypePointer) {
            storageFile = reinterpret_cast<TypePointer const*>(type->second)->getStorageFile();
            // If we have an immediate, which are stored in const memory,
            // inline it
            if (storageFile == SpirvFile::CONST && op->second.storageFile == SpirvFile::IMMEDIATE)
               storageFile = SpirvFile::IMMEDIATE;

            // If we bitcast a pointer to regs to another pointer to regs, keep
            // the old type which has more info.
            if (op->second.type->getType() == spv::Op::OpTypePointer && op->second.storageFile == SpirvFile::TEMPORARY)
               resType = op->second.type;
         }

         spvValues.emplace(resId, SpirVValue{ storageFile, resType, op->second.value, op->second.paddings });
      }
      break;
   case spv::Op::OpCopyMemory: // FALLTHROUGH
   case spv::Op::OpCopyMemorySized:
      {
         const auto targetId = getWord<spv::Id>(firstWord);
         const auto sourceId = getWord<spv::Id>(firstWord + 1u);
         const auto access = ((opcode == spv::Op::OpCopyMemory && numWords > 3u) || (opcode == spv::Op::OpCopyMemorySized && numWords > 4u)) ? getWord<spv::MemoryAccessMask>(firstWord + 2u + (opcode == spv::Op::OpCopyMemorySized)) : spv::MemoryAccessMask::MaskNone;
         const auto alignment = hasFlag(access, spv::MemoryAccessShift::Aligned) ? getWord<uint32_t>(firstWord + 4u) : 1u;

         auto target = spvValues.find(targetId);
         if (target == spvValues.end()) {
            _debug_printf("Couldn't find target with id %u\n", targetId);
            return false;
         }
         auto source = spvValues.find(sourceId);
         if (source == spvValues.end()) {
            _debug_printf("Couldn't find source with id %u\n", sourceId);
            return false;
         }
         uint32_t sizeImm = 0u;
         if (opcode == spv::Op::OpCopyMemorySized) {
            const auto sizeId = getWord<spv::Id>(firstWord + 2u);
            auto size = spvValues.find(sizeId);
            if (size == spvValues.end()) {
               _debug_printf("Couldn't find size with id %u\n", sizeId);
               return false;
            }
            assert(size->second.storageFile == SpirvFile::IMMEDIATE);
            sizeImm = (info->target < 0xc0) ? size->second.value[0u].value->reg.data.u32 : size->second.value[0u].value->reg.data.u64;
         } else {
            sizeImm == reinterpret_cast<const TypePointer *>(target->second.type)->getPointedType()->getSize();
         }
         const auto targetStorage = target->second.storageFile;
         const auto sourceStorage = source->second.storageFile;

         const bool is_packed = target->second.is_packed;

         if (targetStorage == SpirvFile::TEMPORARY && (sourceStorage == SpirvFile::TEMPORARY || sourceStorage == SpirvFile::IMMEDIATE)) {
            for (unsigned int i = 0, c = 0u; i < sizeImm && c < target->second.value.size(); i += typeSizeof(target->second.value[c].value->reg.type), ++c) {
               assert(target->second.value[c].value->reg.size == source->second.value[c].value->reg.size);
               i += source->second.paddings[c];
               mkMov(target->second.value[c].value, source->second.value[c].value, source->second.value[c].value->reg.type);
            }
         } else if (targetStorage == SpirvFile::TEMPORARY) {
            // FIXME load to reg of size alignment, then split things up
            for (unsigned int i = 0, c = 0u; i < sizeImm && c < target->second.value.size(); i += alignment, ++c) {
               auto const offsetImm = mkImm(i);
               Value *offset = getScratch(4u);
               mkMov(offset, offsetImm, TYPE_U32);
               mkLoad(target->second.value[c].value->reg.type, target->second.value[c].value, source->second.value[0u].symbol, offset);
            }
         } else if (sourceStorage == SpirvFile::TEMPORARY || sourceStorage == SpirvFile::IMMEDIATE) {
            if (true /* packed */) {
               unsigned int processedSize = 0u, c = 0u;
               PValue storePointer = target->second.value.front();
               std::stack<const Type *> typesStack;
               const Type *pointedType = reinterpret_cast<const TypePointer *>(source->second.type)->getPointedType();
               for (unsigned int i = 0u; i < sizeImm; i += pointedType->getSize())
                  typesStack.push(pointedType);
               while (processedSize < sizeImm && c < source->second.value.size()) {
                  const auto currentType = typesStack.top();
                  Value *object = source->second.value[c].value;
                  typesStack.pop();
                  if (currentType->isCompooundType()) {
                     for (unsigned int i = currentType->getElementsNb(); i > 0u; --i)
                        typesStack.push(currentType->getElementType(i - 1u));
                     continue;
                  }
                  const auto objectType = object->reg.type;
                  const auto typeSize = (objectType != TYPE_NONE) ? typeSizeof(objectType) : currentType->getSize();

                  // XXX temporary, should not be used with packed
                  processedSize += source->second.paddings[c];

                  auto const split4Byte = [&](Value *value, unsigned int offset){
                     Value *tmp = getScratch(4u);
                     Value *andImm = getScratch(4u);
                     mkMov(andImm, mkImm(0xff), TYPE_U32);
                     mkOp2(OP_AND, TYPE_U32, tmp, value, andImm);
                     store(targetStorage, storePointer, offset + processedSize, tmp, TYPE_U8, access, alignment);
                     Value *shrImm8 = getScratch(4u);
                     mkMov(shrImm8, mkImm(0x8), TYPE_U32);
                     mkOp2(OP_SHR, TYPE_U32, tmp, value, shrImm8);
                     mkOp2(OP_AND, TYPE_U32, tmp, tmp, andImm);
                     store(targetStorage, storePointer, offset + processedSize + 1u, tmp, TYPE_U8, access, alignment);
                     Value *shrImm10 = getScratch(4u);
                     mkMov(shrImm10, mkImm(0x10), TYPE_U32);
                     mkOp2(OP_SHR, TYPE_U32, tmp, value, shrImm10);
                     mkOp2(OP_AND, TYPE_U32, tmp, tmp, andImm);
                     store(targetStorage, storePointer, offset + processedSize + 2u, tmp, TYPE_U8, access, alignment);
                     Value *shrImm18 = getScratch(4u);
                     mkMov(shrImm18, mkImm(0x18), TYPE_U32);
                     mkOp2(OP_SHR, TYPE_U32, tmp, value, shrImm18);
                     store(targetStorage, storePointer, offset + processedSize + 3u, tmp, TYPE_U8, access, alignment);
                  };

                  if (typeSize == 1u) {
                     store(targetStorage, storePointer, processedSize, object, TYPE_U8, access, alignment);
                  } else if (typeSize == 2u) {
                     Value *tmp = getScratch(4u);
                     Value *andImm = getScratch(4u);
                     mkMov(andImm, mkImm(0xff), TYPE_U32);
                     mkOp2(OP_AND, TYPE_U32, tmp, object, andImm);
                     store(targetStorage, storePointer, processedSize, tmp, TYPE_U8, access, alignment);
                     Value *shrImm = getScratch(4u);
                     mkMov(shrImm, mkImm(0x8), TYPE_U32);
                     mkOp2(OP_SHR, TYPE_U32, tmp, object, shrImm);
                     store(targetStorage, storePointer, processedSize + 1u, tmp, TYPE_U8, access, alignment);
                  } else if (typeSize == 4u) {
                     split4Byte(object, 0u);
                  } else if (typeSize == 8u) {
                     Value *splits[2];
                     mkSplit(splits, 4u, object);
                     split4Byte(splits[0], 0u);
                     split4Byte(splits[1], 4u);
                  } else {
                     assert(false);
                  }
                  processedSize += typeSize;
                  ++c;
               }
            } else {
               // TODO
            }
         } else {
            _debug_printf("Unsupported copy setup\n");
            return false;
         }
      }
      break;
   case spv::Op::OpIEqual:
   case spv::Op::OpFOrdEqual:
   case spv::Op::OpINotEqual:
   case spv::Op::OpFOrdNotEqual:
   case spv::Op::OpSGreaterThan:
   case spv::Op::OpUGreaterThan:
   case spv::Op::OpFOrdGreaterThan:
   case spv::Op::OpFUnordGreaterThan:
   case spv::Op::OpSGreaterThanEqual:
   case spv::Op::OpUGreaterThanEqual:
   case spv::Op::OpFOrdGreaterThanEqual:
   case spv::Op::OpFUnordGreaterThanEqual:
   case spv::Op::OpSLessThan:
   case spv::Op::OpULessThan:
   case spv::Op::OpFOrdLessThan:
   case spv::Op::OpFUnordLessThan:
   case spv::Op::OpSLessThanEqual:
   case spv::Op::OpULessThanEqual:
   case spv::Op::OpFOrdLessThanEqual:
   case spv::Op::OpFUnordLessThanEqual:
      if (numWords != 5u) {
         _debug_printf("OpCompositeExtract expects 5 operands but got %u\n", numWords);
         return false;
      }
      {
         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto op1Id = getWord<spv::Id>(firstWord + 2u);
         auto op2Id = getWord<spv::Id>(firstWord + 3u);

         auto op1 = getOp(op1Id);
         if (op1.isUndefined()) {
            _debug_printf("Couldn't find op1 with id %u\n", op1Id);
            return false;
         }
         auto op2 = getOp(op2Id);
         if (op2.isUndefined()) {
            _debug_printf("Couldn't find op2 with id %u\n", op2Id);
            return false;
         }
         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }

         auto op1TypeSearch = spvValues.find(op1Id);
         if (op1TypeSearch == spvValues.end()) {
            _debug_printf("Couldn't fint type for id %u\n", op1Id);
            return false;
         }
         auto op1Type = op1TypeSearch->second.type;
         auto op2TypeSearch = spvValues.find(op2Id);
         if (op2TypeSearch == spvValues.end()) {
            _debug_printf("Couldn't fint type for id %u\n", op2Id);
            return false;
         }
         auto op2Type = op2TypeSearch->second.type;

         if (opcode == spv::Op::OpFOrdEqual ||
             opcode == spv::Op::OpFOrdNotEqual ||
             opcode == spv::Op::OpFOrdGreaterThan ||
             opcode == spv::Op::OpFOrdGreaterThanEqual ||
             opcode == spv::Op::OpFOrdLessThan ||
             opcode == spv::Op::OpFOrdLessThanEqual) {
         } else {
         }
         if (op1Type->getElementsNb() != op2Type->getElementsNb()) {
            _debug_printf("op1 with id %u, and op2 with id %u, should have the same number of elements\n", op1Id, op2Id);
            return false;
         }
         if (op1Type->getElementsNb() != type->second->getElementsNb()) {
            _debug_printf("op1 with id %u, and result type with id %u, should have the same number of elements\n", op1Id, typeId);
            return false;
         }

         int isSrcSigned = -1;
         switch (opcode) {
         case spv::Op::OpSGreaterThan:
         case spv::Op::OpSGreaterThanEqual:
         case spv::Op::OpSLessThan:
         case spv::Op::OpSLessThanEqual:
            isSrcSigned = 1;
            break;
         case spv::Op::OpUGreaterThan:
         case spv::Op::OpUGreaterThanEqual:
         case spv::Op::OpULessThan:
         case spv::Op::OpULessThanEqual:
            isSrcSigned = 0;
            break;
         default:
            break;
         }

         auto pred = getScratch(1, FILE_PREDICATE);
         mkCmp(OP_SET, convertCc(opcode), TYPE_U32, pred, op1Type->getEnumType(isSrcSigned), op1.value, op2.value);
         spvValues.emplace(resId, SpirVValue{ SpirvFile::PREDICATE, type->second, { pred }, type->second->getPaddings() });
      }
      break;
   case spv::Op::OpSNegate:
   case spv::Op::OpFNegate:
      if (numWords != 4u) {
         _debug_printf("Opv1 %u expects 4 operands but got %u\n", numWords);
         return false;
      }
      {
         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto opId = getWord<spv::Id>(firstWord + 2u);

         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }

         auto operation = convertOp(opcode);

         auto value = std::vector<PValue>();
         if (type->second->getElementsNb() == 1u) {
            auto op = getOp(opId, 0u);
            if (op.isUndefined())
               return false;

            auto *tmp = mkOp1v(operation, type->second->getEnumType(), getScratch(op.value->reg.size), op.value);
            value.push_back(tmp);
         } else {
            for (unsigned int i = 0u; i < type->second->getElementsNb(); ++i) {
               auto op = getOp(opId, i + 1u);
               if (op.isUndefined())
                  return false;

               auto *tmp = mkOp1v(operation, type->second->getElementEnumType(i), getScratch(op.value->reg.size), op.value);
               value.push_back(tmp);
            }
         }

         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type->second, value, type->second->getPaddings() });
      }
      break;
   case spv::Op::OpIAdd:
   case spv::Op::OpFAdd:
   case spv::Op::OpISub:
   case spv::Op::OpFSub:
   case spv::Op::OpIMul:
   case spv::Op::OpFMul:
   case spv::Op::OpSDiv:
   case spv::Op::OpUDiv:
   case spv::Op::OpFDiv:
   case spv::Op::OpSMod:
   case spv::Op::OpUMod:
   case spv::Op::OpFMod:
      if (numWords != 5u) {
         _debug_printf("Opv2 %u expects 5 operands but got %u\n", numWords);
         return false;
      }
      {
         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto op1Id = getWord<spv::Id>(firstWord + 2u);
         auto op2Id = getWord<spv::Id>(firstWord + 3u);

         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }

         auto op1TypeSearch = spvValues.find(op1Id);
         if (op1TypeSearch == spvValues.end()) {
            _debug_printf("Couldn't fint type for id %u\n", op1Id);
            return false;
         }
         auto op1Type = op1TypeSearch->second.type;
         auto op2TypeSearch = spvValues.find(op2Id);
         if (op2TypeSearch == spvValues.end()) {
            _debug_printf("Couldn't fint type for id %u\n", op2Id);
            return false;
         }
         auto op2Type = op2TypeSearch->second.type;

         if (opcode == spv::Op::OpFAdd ||
             opcode == spv::Op::OpFSub ||
             opcode == spv::Op::OpFMul ||
             opcode == spv::Op::OpFDiv ||
             opcode == spv::Op::OpFMod) {
         } else {
         }
         if (op1Type->getElementsNb() != op2Type->getElementsNb()) {
            _debug_printf("op1 with id %u, and op2 with id %u, should have the same number of elements\n", op1Id, op2Id);
            return false;
         }
         if (op1Type->getElementsNb() != type->second->getElementsNb()) {
            _debug_printf("op1 with id %u, and result type with id %u, should have the same number of elements\n", op1Id, typeId);
            return false;
         }

         int isSrcSigned = -1;
         switch (opcode) {
         case spv::Op::OpSDiv:
         case spv::Op::OpSMod:
            isSrcSigned = 1;
            break;
         case spv::Op::OpUDiv:
         case spv::Op::OpUMod:
            isSrcSigned = 0;
            break;
         default:
            break;
         }

         auto op = convertOp(opcode);

         auto value = std::vector<PValue>();
         if (type->second->getElementsNb() == 1u) {
            auto op1 = getOp(op1Id, 0u);
            if (op1.isUndefined()) {
               _debug_printf("Couldn't find op1 with id %u\n", op1Id);
               return false;
            }
            auto op2 = getOp(op2Id, 0u);
            if (op2.isUndefined()) {
               _debug_printf("Couldn't find op2 with id %u\n", op2Id);
               return false;
            }

            auto *tmp = mkOp2v(op, type->second->getEnumType(isSrcSigned), getScratch(op1.value->reg.size), op1.value, op2.value);
            value.push_back(tmp);
         } else {
            for (unsigned int i = 0u; i < type->second->getElementsNb(); ++i) {
               auto op1 = getOp(op1Id, i);
               if (op1.isUndefined()) {
                  _debug_printf("Couldn't find component %u for op1 with id %u\n", i, op1Id);
                  return false;
               }
               auto op2 = getOp(op2Id, i);
               if (op2.isUndefined()) {
                  _debug_printf("Couldn't find component %u for op2 with id %u\n", i, op2Id);
                  return false;
               }

               auto *tmp = mkOp2v(op, type->second->getElementEnumType(i, isSrcSigned), getScratch(op1.value->reg.size), op1.value, op2.value);
               value.push_back(tmp);
            }
         }

         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type->second, value, type->second->getPaddings() });
      }
      break;
   case spv::Op::OpSRem:
   case spv::Op::OpFRem:
      if (numWords != 5u) {
         _debug_printf("Opv2 %u expects 5 operands but got %u\n", numWords);
         return false;
      }
      {
         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto op1Id = getWord<spv::Id>(firstWord + 2u);
         auto op2Id = getWord<spv::Id>(firstWord + 3u);

         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }

         auto op1TypeSearch = spvValues.find(op1Id);
         if (op1TypeSearch == spvValues.end()) {
            _debug_printf("Couldn't fint type for id %u\n", op1Id);
            return false;
         }
         auto op1Type = op1TypeSearch->second.type;
         auto op2TypeSearch = spvValues.find(op2Id);
         if (op2TypeSearch == spvValues.end()) {
            _debug_printf("Couldn't fint type for id %u\n", op2Id);
            return false;
         }
         auto op2Type = op2TypeSearch->second.type;

         if (opcode == spv::Op::OpFAdd ||
             opcode == spv::Op::OpFSub ||
             opcode == spv::Op::OpFMul ||
             opcode == spv::Op::OpFDiv ||
             opcode == spv::Op::OpFMod) {
         } else {
         }
         if (op1Type->getElementsNb() != op2Type->getElementsNb()) {
            _debug_printf("op1 with id %u, and op2 with id %u, should have the same number of elements\n", op1Id, op2Id);
            return false;
         }
         if (op1Type->getElementsNb() != type->second->getElementsNb()) {
            _debug_printf("op1 with id %u, and result type with id %u, should have the same number of elements\n", op1Id, typeId);
            return false;
         }

         int isSrcSigned = -1;
         switch (opcode) {
         case spv::Op::OpSRem:
            isSrcSigned = 1;
            break;
         default:
            break;
         }

         auto value = std::vector<PValue>();
         if (type->second->getElementsNb() == 1u) {
            auto op1 = getOp(op1Id, 0u).value;
            if (op1 == nullptr) {
               _debug_printf("Couldn't find op1 with id %u\n", op1Id);
               return false;
            }
            auto op2 = getOp(op2Id, 0u).value;
            if (op2 == nullptr) {
               _debug_printf("Couldn't find op2 with id %u\n", op2Id);
               return false;
            }

            auto *tmp1 = mkOp2v(OP_DIV, type->second->getEnumType(isSrcSigned), getScratch(op1->reg.size), op1, op2);
            auto *tmp2 = mkOp2v(OP_MUL, type->second->getEnumType(isSrcSigned), getScratch(op1->reg.size), tmp1, op2);
            auto *tmpRes = mkOp2v(OP_SUB, type->second->getEnumType(isSrcSigned), getScratch(op1->reg.size), op1, tmp2);
            value.push_back(tmpRes);
         } else {
            for (unsigned int i = 0u; i < type->second->getElementsNb(); ++i) {
               auto op1 = getOp(op1Id, i).value;
               if (op1 == nullptr) {
                  _debug_printf("Couldn't find composante %u for op1 with id %u\n", i, op1Id);
                  return false;
               }
               auto op2 = getOp(op2Id, i).value;
               if (op2 == nullptr) {
                  _debug_printf("Couldn't find composante %u op2 with id %u\n", i, op2Id);
                  return false;
               }

               auto *tmp1 = mkOp2v(OP_DIV, type->second->getElementEnumType(i, isSrcSigned), getScratch(op1->reg.size), op1, op2);
               auto *tmp2 = mkOp2v(OP_MUL, type->second->getElementEnumType(i, isSrcSigned), getScratch(op1->reg.size), op2, tmp1);
               auto *tmpRes = mkOp2v(OP_SUB, type->second->getElementEnumType(i, isSrcSigned), getScratch(op1->reg.size), op1, tmp2);
               value.push_back(tmpRes);
            }
         }

         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type->second, value, type->second->getPaddings() });
      }
      break;
   case spv::Op::OpAtomicExchange:
   case spv::Op::OpAtomicIIncrement:
   case spv::Op::OpAtomicIDecrement:
   case spv::Op::OpAtomicIAdd:
   case spv::Op::OpAtomicISub:
   case spv::Op::OpAtomicSMin:
   case spv::Op::OpAtomicUMin:
   case spv::Op::OpAtomicSMax:
   case spv::Op::OpAtomicUMax:
   case spv::Op::OpAtomicAnd:
   case spv::Op::OpAtomicOr:
   case spv::Op::OpAtomicXor:
      {
         auto const has_no_value = (opcode == spv::Op::OpAtomicIIncrement) || (opcode == spv::Op::OpAtomicIDecrement);
         if (has_no_value) {
            if (numWords != 6u) {
               _debug_printf("OpAtomic(IIncrement|IDecrement) %u expects 6 operands but got %u\n", opcode, numWords);
               return false;
            }
         } else {
            if (numWords != 7u) {
               _debug_printf("OpAtomic(Exchange|IAdd|ISub|Smin|UMin|SMax|UMax|And|Or|Xor) %u expects 7 operands but got %u\n", opcode, numWords);
               return false;
            }
         }

         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto pointerId = getWord<spv::Id>(firstWord + 2u);
         auto scope = getWord<spv::Scope>(firstWord + 3u);
         auto memorySemantics = getWord<spv::MemorySemanticsMask>(firstWord + 4u);
         auto valueId = has_no_value ? 0u : getWord<spv::Id>(firstWord + 5u);

         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }

         auto values = std::vector<PValue>();
         auto pointer = getOp(pointerId, 0u).value; // Will that still work?
         if (pointer == nullptr) {
            _debug_printf("Couldn't find pointer with id %u\n", pointerId);
            return false;
         }
         auto tmp_value = has_no_value ? nullptr : getOp(valueId, 0u).value;
         if (tmp_value == nullptr && !has_no_value) {
            _debug_printf("Couldn't find value with id %u\n", valueId);
            return false;
         }

         int isSrcSigned = -1;
         switch (opcode) {
         case spv::Op::OpAtomicSMin:
         case spv::Op::OpAtomicSMax:
            isSrcSigned = 1;
            break;
         case spv::Op::OpAtomicUMin:
         case spv::Op::OpAtomicUMax:
            isSrcSigned = 0;
            break;
         default:
            break;
         }

         Value* value = nullptr;
         if (opcode == spv::Op::OpAtomicIDecrement) {
            value = getScratch(type->second->getSize());
            mkMov(value, mkImm(-1), type->second->getEnumType(isSrcSigned));
         } else if (opcode == spv::Op::OpAtomicISub) {
            value = getScratch(type->second->getSize());
            mkOp2(OP_SUB, type->second->getEnumType(isSrcSigned), value, mkImm(0), tmp_value);
         } else {
            value = tmp_value;
         }

         auto tmp = getScratch(type->second->getSize());
         auto base = acquire(SpirvFile::GLOBAL, spvValues.find(pointerId)->second.type);
         auto insn = opcode == spv::Op::OpAtomicIIncrement ? mkOp1(OP_ATOM, type->second->getEnumType(isSrcSigned), tmp, base) : mkOp2(OP_ATOM, type->second->getEnumType(isSrcSigned), tmp, base, value);
         insn->subOp = getSubOp(opcode);
         insn->setIndirect(0, 0, pointer);
         if (opcode == spv::Op::OpAtomicISub)
            insn->src(1).mod.neg();
         values.push_back(tmp);

         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type->second, values, type->second->getPaddings() });
      }
      break;
   case spv::Op::OpAtomicCompareExchange:
      {
         if (numWords != 9u) {
            _debug_printf("OpAtomicCompareExchange expects 9 operands but got %u\n", numWords);
            return false;
         }

         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto pointerId = getWord<spv::Id>(firstWord + 2u);
         auto scope = getWord<spv::Scope>(firstWord + 3u);
         auto memorySemanticsEqual = getWord<spv::MemorySemanticsMask>(firstWord + 4u);
         auto memorySemanticsUnequal = getWord<spv::MemorySemanticsMask>(firstWord + 5u);
         auto valueId = getWord<spv::Id>(firstWord + 6u);
         auto comparatorId = getWord<spv::Id>(firstWord + 7u);

         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }

         auto values = std::vector<PValue>();
         auto pointer = getOp(pointerId, 0u).value;
         if (pointer == nullptr) {
            _debug_printf("Couldn't find pointer with id %u\n", pointerId);
            return false;
         }
         auto value = getOp(valueId, 0u).value;
         if (value == nullptr) {
            _debug_printf("Couldn't find value with id %u\n", valueId);
            return false;
         }
         auto comparator = getOp(comparatorId, 0u).value;
         if (comparator == nullptr) {
            _debug_printf("Couldn't find comparator with id %u\n", comparatorId);
            return false;
         }

         auto tmp = getScratch(type->second->getSize());
         auto base = acquire(SpirvFile::GLOBAL, spvValues.find(pointerId)->second.type);
         auto insn = mkOp3(OP_ATOM, type->second->getEnumType(), tmp, base, value, comparator);
         insn->subOp = getSubOp(opcode);
         insn->setIndirect(0, 0, pointer);
         values.push_back(tmp);

         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type->second, values, type->second->getPaddings() });
      }
      break;
   case spv::Op::OpVectorTimesScalar:
      if (numWords != 5u) {
         _debug_printf("OpVectorTimesScalar %u expects 5 operands but got %u\n", numWords);
         return false;
      }
      {
         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto op1Id = getWord<spv::Id>(firstWord + 2u);
         auto op2Id = getWord<spv::Id>(firstWord + 3u);

         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }

         auto value = std::vector<PValue>();
         auto op2 = getOp(op2Id, 0u).value;
         if (op2 == nullptr) {
            _debug_printf("Couldn't find op2 with id %u\n", op2Id);
            return false;
         }
         for (unsigned int i = 0u; i < type->second->getElementsNb(); ++i) {
            auto op1 = getOp(op1Id, i).value;
            if (op1 == nullptr) {
               _debug_printf("OpVectorTimesScalar %u: Couldn't find component %u op1 with id %u\n", resId, i, op1Id);
               return false;
            }

            auto *tmp = mkOp2v(OP_MUL, type->second->getElementEnumType(i), getScratch(op1->reg.size), op1, op2);
            value.push_back(tmp);
         }

         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type->second, value, type->second->getPaddings() });
      }
      break;
   case spv::Op::OpUConvert:
   case spv::Op::OpSConvert:
   case spv::Op::OpConvertUToF:
   case spv::Op::OpConvertFToU:
   case spv::Op::OpConvertSToF:
   case spv::Op::OpConvertFToS:
   case spv::Op::OpSatConvertSToU:
   case spv::Op::OpSatConvertUToS:
      if (numWords != 4u) {
         _debug_printf("Op*Convert* %u expects 4 operands but got %u\n", numWords);
         return false;
      }
      {
         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto srcId = getWord<spv::Id>(firstWord + 2u);

         auto src = getOp(srcId).value;
         if (src == nullptr) {
            _debug_printf("Couldn't find src with id %u\n", srcId);
            return false;
         }

         auto type = types.find(typeId);
         if (type == types.end()) {
            _debug_printf("Couldn't find type with id %u\n", typeId);
            return false;
         }

         auto srcType = getType(srcId);
         if (srcType == nullptr) {
             _debug_printf("Couldn't find type for id %u\n", srcId);
             return false;
         }

         int isSrcSigned = -1;
         int isDstSigned = -1;
         switch (opcode) {
         case spv::Op::OpUConvert:
            isSrcSigned = 0;
            isDstSigned = 0;
            break;
         case spv::Op::OpSConvert:
            isSrcSigned = 1;
            isDstSigned = 1;
            break;
         case spv::Op::OpConvertUToF:
            isSrcSigned = 0;
            isDstSigned = 1;
            break;
         case spv::Op::OpConvertFToU:
            isSrcSigned = 1;
            isDstSigned = 0;
            break;
         case spv::Op::OpConvertSToF:
         case spv::Op::OpConvertFToS:
            isSrcSigned = 1;
            isDstSigned = 1;
            break;
         case spv::Op::OpSatConvertUToS:
            isSrcSigned = 0;
            isDstSigned = 1;
            break;
         case spv::Op::OpSatConvertSToU:
            isSrcSigned = 1;
            isDstSigned = 0;
            break;
         default:
            assert(false && "Unsupported opcode");
            break;
         }

         int saturate = opcode == spv::Op::OpSatConvertSToU || opcode == spv::Op::OpSatConvertUToS;

         Value *res = nullptr;
         if (type->second->getSize() >= 4u && srcType->getSize() >= 4u) {
            // FIXME doesn't work for vectors
            res = getScratch(type->second->getSize());
            mkCvt(OP_CVT, type->second->getEnumType(isDstSigned), res, srcType->getEnumType(isSrcSigned), src)->saturate = saturate;
         } else {
            res = src;
         }

         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type->second, { res }, type->second->getPaddings() });
      }
      break;
   case spv::Op::OpSampledImage:
      {
         if (numWords != 5u) {
            _debug_printf("OpSampledImage expects 5 operands but got %y\n", numWords);
            return false;
         }

         auto typeId = getWord<spv::Id>(firstWord);
         auto resId = getWord<spv::Id>(firstWord + 1u);
         auto imageId = getWord<spv::Id>(firstWord + 2u);
         auto samplerId = getWord<spv::Id>(firstWord + 3u);

         auto searchType = types.find(typeId);
         if (searchType == types.end()) {
            _debug_printf("Could not find type %u\n", typeId);
            return false;
         }
         auto searchImage = spvValues.find(imageId);
         if (searchImage == spvValues.end()) {
            _debug_printf("Could not find image %u\n", imageId);
            return false;
         }
         auto searchSampler = samplers.find(samplerId);
         if (searchSampler == samplers.end()) {
            _debug_printf("Could not find sampler %u\n", samplerId);
            return false;
         }

         sampledImages.emplace(resId, SampledImage{ reinterpret_cast<TypeSampledImage const*>(searchType->second), searchImage->second.value.front().value, searchSampler->second });
      }
      break;
   case spv::Op::OpImageSampleExplicitLod:
      {
         if (numWords < 7u) {
            _debug_printf("OpImageSampleExplicitLod expects at least 7 operands but got %u\n", numWords);
            return false;
         }

         auto const typeId = getWord<spv::Id>(firstWord);
         auto const resId = getWord<spv::Id>(firstWord + 1u);
         auto const sampledImageId = getWord<spv::Id>(firstWord + 2u);
         auto const coordinatesId = getWord<spv::Id>(firstWord + 3u);
         auto const operand = getWord<spv::ImageOperandsMask>(firstWord + 4u);
         auto operandArgs = std::vector<spv::Id>();
         for (unsigned int i = firstWord + 5u; i < numWords; ++i)
            operandArgs.push_back(getWord<spv::Id>(i));

         auto searchType = types.find(typeId);
         if (searchType == types.end()) {
            _debug_printf("Could not find type %u\n", typeId);
            return false;
         }
         auto searchSampledImage = sampledImages.find(sampledImageId);
         if (searchSampledImage == sampledImages.end()) {
            _debug_printf("Could not find sampled image %u\n", sampledImageId);
            return false;
         }
         auto searchCoordinates = spvValues.find(coordinatesId);
         if (searchCoordinates == spvValues.end()) {
            _debug_printf("Could not find sampler %u\n", coordinatesId);
            return false;
         }

         auto const componentSize = searchType->second->getElementType(0u)->getSize();
         std::vector<PValue> res = { getScratch(componentSize), getScratch(componentSize), getScratch(componentSize), getScratch(componentSize) };
         std::vector<Value*> resValue;
         for (auto &i : res)
            resValue.push_back(i.value);
         // TODO
         auto const sampledImageType = reinterpret_cast<TypeSampledImage const*>(searchSampledImage->second.type);
         auto const imageTypeId = sampledImageType->getImageType();
         auto searchImageType = types.find(imageTypeId);
         if (searchImageType == types.end()) {
            _debug_printf("Could not find type %u for sampler type %u\n", imageTypeId, sampledImageId);
            return false;
         }
         auto const imageTarget = getTexTarget(reinterpret_cast<TypeImage const*>(searchImageType->second));
         auto const tic = 0;
         std::vector<Value*> args;
         for (auto &i : searchCoordinates->second.value)
            args.push_back(i.value);
         auto ld = mkTex(OP_SULDP, imageTarget, tic, 0, resValue, args);
         ld->tex.mask = 0;
         ld->tex.format = getImageFormat(reinterpret_cast<TypeImage const*>(searchImageType->second)->format);
         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, searchType->second, res, { 1u } });
      }
      break;
   case spv::Op::OpImageWrite:
      {
         if (numWords < 4u) {
            _debug_printf("OpImageWrite expects at least 4 operands but got %u\n", numWords);
            return false;
         }

         auto const imageId = getWord<spv::Id>(firstWord);
         auto const coordinatesId = getWord<spv::Id>(firstWord + 1u);
         auto const operand = getWord<spv::ImageOperandsMask>(firstWord + 2u);
         auto operandArgs = std::vector<spv::Id>();
         for (unsigned int i = firstWord + 3u; i < numWords; ++i)
            operandArgs.push_back(getWord<spv::Id>(i));

         auto searchImage = spvValues.find(imageId);
         if (searchImage == spvValues.end()) {
            _debug_printf("Could not find image %u\n", imageId);
            return false;
         }
         auto searchCoordinates = spvValues.find(coordinatesId);
         if (searchCoordinates == spvValues.end()) {
            _debug_printf("Could not find sampler %u\n", coordinatesId);
            return false;
         }

         // TODO
         auto const imageTarget = getTexTarget(reinterpret_cast<TypeImage const*>(searchImage->second.type));
         auto const tic = 0;
         std::vector<Value*> args;
         for (auto &i : searchCoordinates->second.value)
            args.push_back(i.value);
         auto st = mkTex(OP_SUSTP, imageTarget, tic, 0, {}, args);
         st->tex.mask = TGSI_WRITEMASK_XY;
         st->tex.format = getImageFormat(reinterpret_cast<TypeImage const*>(searchImage->second.type)->format);
//         st->cache = tgsi.getCacheMode();
      }
      break;
   default:
      return false;
   }

   return true;
}

bool
Converter::convertEntryPoint(unsigned int numWords, unsigned int firstWord)
{
   if (numWords < 4u) {
      _debug_printf("OpEntryPoint expects at least 4 operands but got %u\n", numWords);
      return false;
   }

   std::string name = binary + (firstWord + 2u) * sizeof(uint32_t);
   unsigned int nextOperand = name.size() / 4u + (name.size() % 4u) != 0u;
   std::vector<spv::Id> references = std::vector<spv::Id>();
   for (unsigned int i = nextOperand; i < firstWord + numWords; ++i)
      references.push_back(getWord<spv::Id>(i));
   EntryPoint entryPoint = { static_cast<uint32_t>(entryPoints.size()), getWord<spv::ExecutionModel>(firstWord),
                             name, references
   };
   spv::Id id = getWord<spv::Id>(firstWord + 1u);
   entryPoints.emplace(id, entryPoint);
   auto search = names.find(id);
   if (search == names.end())
      names.emplace(id, name);

   return true;
}

bool
Converter::convertDecorate(unsigned int numWords, unsigned int firstWord,
                           bool hasMember)
{
   assert(!hasMember);
   unsigned int offset = static_cast<unsigned int>(hasMember);
   if (numWords < (3u + offset)) {
      _debug_printf("OpDecorate expects at least %u operands but got %u\n", (3u + offset), numWords);
      return false;
   }

   Words literals = Words();
   for (unsigned int i = 2u + offset; i < numWords - 1u; ++i)
      literals.push_back(getWord<unsigned>(firstWord + i));
   decorations[getWord<spv::Id>(firstWord)][getWord<spv::Decoration>(firstWord + 1u + offset)].emplace_back(literals);

   return true;
}

bool
Converter::loadBuiltin(spv::Id dstId, Type const* dstType, Words const& decLiterals, spv::MemoryAccessMask access)
{
   auto const builtin = static_cast<spv::BuiltIn>(decLiterals[0u]);

   auto const& type = dstType->getElementType(0u);
   auto const typeEnum = type->getEnumType();
   auto const typeSize = type->getSize();
   auto getTid = [&](unsigned int id){
      auto tidSysval = mkSysVal(SV_TID, id);
      auto tidReg = getScratch(tidSysval->reg.size);
      mkOp1(OP_RDSV, tidSysval->reg.type, tidReg, tidSysval);
      if (tidSysval->reg.type != typeEnum) {
         auto res = getScratch(typeSize);
         mkCvt(OP_CVT, typeEnum, res, tidSysval->reg.type, tidReg);
         return res;
      } else {
         return tidReg;
      }

   };
   auto getNtid = [&](unsigned int id){
      auto ntidSysval = mkSysVal(SV_NTID, id);
      auto ntidReg = getScratch(ntidSysval->reg.size);
      mkOp1(OP_RDSV, ntidSysval->reg.type, ntidReg, ntidSysval);
      if (ntidSysval->reg.type != typeEnum) {
         auto res = getScratch(typeSize);
         mkCvt(OP_CVT, typeEnum, res, ntidSysval->reg.type, ntidReg);
         return res;
      } else {
         return ntidReg;
      }
   };
   auto getGid = [&](unsigned int index){
      auto tidSysval = mkSysVal(SV_TID, index);
      auto ntidSysval = mkSysVal(SV_NTID, index);
      auto ctaidSysval = mkSysVal(SV_CTAID, index);

      auto sysValType = tidSysval->reg.type;
      auto sysValSize = tidSysval->reg.size;
      assert(sysValType == ntidSysval->reg.type && sysValType == ctaidSysval->reg.type);

      auto tid    = mkOp1v(OP_RDSV, sysValType, getScratch(sysValSize), tidSysval);
      auto ntid   = mkOp1v(OP_RDSV, sysValType, getScratch(sysValSize), ntidSysval);
      auto ctaid  = mkOp1v(OP_RDSV, sysValType, getScratch(sysValSize), ctaidSysval);
      auto tmp    = mkOp2v(OP_ADD,  sysValType, getScratch(sysValSize), tid, mkImm(0u)); // FIXME should be GID_OFF
      auto tmpRes = mkOp3v(OP_MAD,  sysValType, getScratch(sysValSize), ntid, ctaid, tmp);
      if (sysValType != typeEnum) {
         auto res = getScratch(typeSize);
         mkCvt(OP_CVT, typeEnum, res, sysValType, tmpRes);
         return res;
      } else {
         return tmpRes;
      }
   };
   auto getNGid = [&](unsigned int index){
      auto ntidSysval = mkSysVal(SV_NTID, index);
      auto nctaidSysval = mkSysVal(SV_NCTAID, index);

      auto sysValType = ntidSysval->reg.type;
      auto sysValSize = ntidSysval->reg.size;
      assert(sysValType == nctaidSysval->reg.type);

      auto ntid   = mkOp1v(OP_RDSV, sysValType, getScratch(sysValSize), ntidSysval);
      auto nctaid = mkOp1v(OP_RDSV, sysValType, getScratch(sysValSize), nctaidSysval);
      auto tmp    = mkOp2v(OP_MUL,  sysValType, getScratch(sysValSize), ntid, nctaid);
      if (sysValType != typeEnum) {
         auto res = getScratch(typeSize);
         mkCvt(OP_CVT, typeEnum, res, sysValType, tmp);
         return res;
      } else {
         return tmp;
      }
   };

   std::function<Value *(unsigned int)> vec3Func;
   switch (builtin) {
   case spv::BuiltIn::LocalInvocationId:
      vec3Func = getTid;
      break;
   case spv::BuiltIn::WorkgroupSize:
      vec3Func = getNtid;
      break;
   case spv::BuiltIn::GlobalInvocationId:
      vec3Func = getGid;
      break;
   case spv::BuiltIn::GlobalSize:
      vec3Func = getNGid;
      break;
   default:
      break;
   }
   switch (builtin) {
   case spv::BuiltIn::LocalInvocationId:
   case spv::BuiltIn::WorkgroupSize:
   case spv::BuiltIn::GlobalInvocationId:
   case spv::BuiltIn::GlobalSize:
      {
         if (!dstType->isVectorOfSize(3u) || !dstType->getElementType(0u)->isUInt()) {
            _debug_printf("Builtin %u should be a vector of 3 uint\n", builtin);
            return false;
         }
         auto value = std::vector<PValue>{ vec3Func(0u), vec3Func(1u), vec3Func(2u) };
         spvValues.emplace(dstId, SpirVValue{ SpirvFile::TEMPORARY, dstType, value, { 1u, 1u, 1u } });
      }
      break;
   default:
      _debug_printf("Unsupported builtin %u\n", builtin);
      return false;
   }

   return true;
}

bool
Converter::convertOpenCLInstruction(spv::Id resId, Type const* type, uint32_t op, uint32_t firstWord, uint32_t numWords)
{
   auto getOp = [&](spv::Id id, unsigned c = 0u){
      auto searchOp = spvValues.find(id);
      if (searchOp == spvValues.end())
         return static_cast<Value*>(nullptr);

      auto& opStruct = searchOp->second;
      if (c >= opStruct.value.size()) {
         _debug_printf("Trying to access element %u out of %u\n", c, opStruct.value.size());
         return static_cast<Value*>(nullptr);
      }

      auto op = opStruct.value[c].value;
      if (opStruct.storageFile == SpirvFile::IMMEDIATE) {
         auto constant = op;
         op = getScratch(constant->reg.size);
         mkMov(op, constant, constant->reg.type);
      }
      return op;
   };

   switch (op) {
   case 167:
   case 168:
      {
         if (numWords != 8) {
            _debug_printf("OpExtInst:(s|u)_mad24 expects 8 operands but got %u\n", numWords);
            return false;
         }
         auto op1 = getOp(getWord<spv::Id>(firstWord + 4u), 0u);
         auto op2 = getOp(getWord<spv::Id>(firstWord + 5u), 0u);
         auto op3 = getOp(getWord<spv::Id>(firstWord + 6u), 0u);
         auto res = getScratch();
         mkOp3(OP_MADSP, type->getEnumType(op == 167), res, op1, op2, op3)->subOp = NV50_IR_SUBOP_MADSP(2, 2, 0); // u24 u24 u32
         spvValues.emplace(resId, SpirVValue{ SpirvFile::TEMPORARY, type, { res }, type->getPaddings() });
         return true;
      }
      break;
   }

   return false;
}

int
Converter::getSubOp(spv::Op opcode) const
{
   switch (opcode) {
   case spv::Op::OpAtomicIIncrement: return NV50_IR_SUBOP_ATOM_INC;
   case spv::Op::OpAtomicIDecrement: return NV50_IR_SUBOP_ATOM_ADD;
   case spv::Op::OpAtomicIAdd: return NV50_IR_SUBOP_ATOM_ADD;
   case spv::Op::OpAtomicISub: return NV50_IR_SUBOP_ATOM_ADD;
   case spv::Op::OpAtomicSMin: /* fallthrough */
   case spv::Op::OpAtomicUMin: return NV50_IR_SUBOP_ATOM_MIN;
   case spv::Op::OpAtomicSMax: /* fallthrough */
   case spv::Op::OpAtomicAnd: return NV50_IR_SUBOP_ATOM_AND;
   case spv::Op::OpAtomicOr: return NV50_IR_SUBOP_ATOM_OR;
   case spv::Op::OpAtomicXor: return NV50_IR_SUBOP_ATOM_XOR;
   case spv::Op::OpAtomicCompareExchange: return NV50_IR_SUBOP_ATOM_CAS;
   case spv::Op::OpAtomicExchange: return NV50_IR_SUBOP_ATOM_EXCH;
   default: assert(false); return 0;
   }
}

TexTarget
Converter::getTexTarget(TypeImage const* type)
{
   switch (type->dim) {
   case spv::Dim::Dim1D:
      if (type->arrayed && type->depth == 1)
         return TexTarget::TEX_TARGET_1D_ARRAY_SHADOW;
      else if (type->arrayed)
         return TexTarget::TEX_TARGET_1D_ARRAY;
      else if (type->depth == 1)
         return TexTarget::TEX_TARGET_1D_SHADOW;
      else
         return TexTarget::TEX_TARGET_1D;
   case spv::Dim::Dim2D:
      if (type->arrayed && type->depth == 1)
         return TexTarget::TEX_TARGET_2D_ARRAY_SHADOW;
      else if (type->arrayed && type->ms)
         return TexTarget::TEX_TARGET_2D_MS_ARRAY;
      else if (type->arrayed)
         return TexTarget::TEX_TARGET_2D_ARRAY;
      else if (type->depth == 1)
         return TexTarget::TEX_TARGET_2D_SHADOW;
      else if (type->ms)
         return TexTarget::TEX_TARGET_2D_MS;
      else
         return TexTarget::TEX_TARGET_2D;
   case spv::Dim::Dim3D:
      return TexTarget::TEX_TARGET_3D;
   case spv::Dim::Rect:
      if (type->depth == 1)
         return TexTarget::TEX_TARGET_RECT_SHADOW;
      else
         return TexTarget::TEX_TARGET_RECT;
   case spv::Dim::Buffer:
      return TexTarget::TEX_TARGET_BUFFER;
   case spv::Dim::Cube:
      if (type->arrayed && type->depth == 1)
         return TexTarget::TEX_TARGET_CUBE_ARRAY_SHADOW;
      if (type->arrayed)
         return TexTarget::TEX_TARGET_CUBE_ARRAY;
      if (type->depth == 1)
         return TexTarget::TEX_TARGET_CUBE_SHADOW;
      else
         return TexTarget::TEX_TARGET_CUBE;
   case spv::Dim::SubpassData:
      assert(false && "Unsupported Dim::SubpassData");
      return TexTarget::TEX_TARGET_1D;
   }
}

TexInstruction::ImgFormatDesc const*
Converter::getImageFormat(spv::ImageFormat format)
{
   ImgFormat imgFormat = ImgFormat::FMT_NONE;

#define NV50_IR_TRANS_IMG_FORMAT(a, b) case a: imgFormat = FMT_##b; break;

   switch (format) {
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Unknown, NONE)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba32f, RGBA32F)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba16f, RGBA16F)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R32f, R32F)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba8, RGBA8)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba8Snorm, RGBA8_SNORM)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg32f, RG32F)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg16f, RG16F)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R11fG11fB10f, R11G11B10F)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R16f, R16F)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba16, RGBA16)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgb10A2, RGB10A2)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg16, RG16)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg8, RG8)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R16, R16)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R8, R8)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba16Snorm, RGBA16_SNORM)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg16Snorm, RG16_SNORM)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg8Snorm, RG8_SNORM)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R16Snorm, R16_SNORM)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R8Snorm, R8_SNORM)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba32i, RGBA32I)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba16i, RGBA16I)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba8i, RGBA8I)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R32i, R32I)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg32i, RG32I)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg16i, RG16I)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg8i, RG8I)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R16i, R16I)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R8i, R8I)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba32ui, RGBA32UI)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba16ui, RGBA16UI)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgba8ui, RGBA8UI)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R32ui, R32UI)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rgb10a2ui, RGB10A2UI)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg32ui, RG32UI)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg16ui, RG16UI)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::Rg8ui, RG8UI)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R16ui, R16UI)
      NV50_IR_TRANS_IMG_FORMAT(spv::ImageFormat::R8ui, R8UI)
   }
   return &nv50_ir::TexInstruction::formatTable[imgFormat];
}

} // unnamed namespace

namespace nv50_ir {

bool
Program::makeFromSPIRV(struct nv50_ir_prog_info *info)
{
   Converter builder(this, info);
   return builder.run();
}

} // namespace nv50_ir
