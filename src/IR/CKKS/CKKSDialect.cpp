#include "heco/IR/CKKS/CKKSDialect.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/TypeSupport.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"

using namespace mlir;
using namespace heco;
using namespace ckks;

//===----------------------------------------------------------------------===//
// TableGen'd Type definitions
//===----------------------------------------------------------------------===//
#define GET_TYPEDEF_CLASSES
#include "heco/IR/CKKS/CKKSTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// TableGen'd Operation definitions
//===----------------------------------------------------------------------===//
#define GET_OP_CLASSES
#include "heco/IR/CKKS/CKKS.cpp.inc"

::mlir::LogicalResult ckks::MultiplyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    // Operand adaptors (https://mlir.llvm.org/docs/OpDefinitions/#operand-adaptors) provide a convenient way to access
    // operands when given as a "generic" triple of ValueRange, DictionaryAttr, RegionRange  instead of nicely
    // "packaged" inside the operation class.
    auto op = MultiplyOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to ckks.multiply must be of type ckks.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to ckks.multiply must have same elementType.");
    auto new_size = (type_x.getSize() - 1) + (type_y.getSize() - 1) + 1;
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::MultiplyManyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = MultiplyManyOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX()[0].getType().dyn_cast<CiphertextType>();
    auto new_size = type_x.getSize();
    for (auto xx : op.getX())
    {
        CiphertextType type_xx = xx.getType().dyn_cast<CiphertextType>();
        assert(type_x && type_xx && "Inputs to ckks.add must be of type ckks.ctxt."); // Should never trigger
        assert(
            type_x.getElementType() == type_xx.getElementType() &&
            "Inputs to ckks.add_many must have same elementType.");
        new_size = std::max(new_size, type_xx.getSize());
    }
    new_size = new_size + 1;
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::SubOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = SubOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to ckks.sub must be of type ckks.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to ckks.sub must have same elementType.");
    auto new_size = std::max(type_x.getSize(), type_y.getSize());
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::AddOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to ckks.add must be of type ckks.ctxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to ckks.add must have same elementType.");
    auto new_size = std::max(type_x.getSize(), type_y.getSize());
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::SigmoidOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    auto size = type_x.getSize();
      inferredReturnTypes.push_back(CiphertextType::get(context, size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::ConvOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ConvOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    CiphertextType type_z = op.getZ().getType().dyn_cast<CiphertextType>();
    //CiphertextType type_conv_k = op.getConvK().getType().dyn_cast<CiphertextType>();
    auto size = type_x.getSize();
      inferredReturnTypes.push_back(CiphertextType::get(context, size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::ReluOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    auto size = type_x.getSize();
      inferredReturnTypes.push_back(CiphertextType::get(context, size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::PoolOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    auto size = type_x.getSize();
      inferredReturnTypes.push_back(CiphertextType::get(context, size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::FlattenOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    auto size = type_x.getSize();
      inferredReturnTypes.push_back(CiphertextType::get(context, size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::FcOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = FcOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    CiphertextType type_z = op.getZ().getType().dyn_cast<CiphertextType>();
    auto size = type_x.getSize();
      inferredReturnTypes.push_back(CiphertextType::get(context, size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::SoftmaxOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    auto size = type_x.getSize();
      inferredReturnTypes.push_back(CiphertextType::get(context, size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::AddManyOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddManyOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX()[0].getType().dyn_cast<CiphertextType>();
    auto new_size = type_x.getSize();
    for (auto xx : op.getX())
    {
        CiphertextType type_xx = xx.getType().dyn_cast<CiphertextType>();
        assert(type_x && type_xx && "Inputs to ckks.add must be of type ckks.ctxt."); // Should never trigger
        assert(
            type_x.getElementType() == type_xx.getElementType() &&
            "Inputs to ckks.add_many must have same elementType.");
        new_size = std::max(new_size, type_xx.getSize());
    }
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::MultiplyPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = MultiplyPlainOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.getY().getType().dyn_cast<PlaintextType>();
    assert(
        type_x && type_y &&
        "Inputs to ckks.multiply_plain must be of type ckks.ctxt & ckks.ptxt."); // Should never trigger
    assert(
        type_x.getElementType() == type_y.getElementType() &&
        "Inputs to ckks.multiply_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::AddPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = AddPlainOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.getY().getType().dyn_cast<PlaintextType>();
    assert(
        type_x && type_y &&
        "Inputs to ckks.multiply_plain must be of type ckks.ctxt & ckks.ptxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to ckks.add_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::SubPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = SubPlainOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    PlaintextType type_y = op.getY().getType().dyn_cast<PlaintextType>();
    assert(
        type_x && type_y &&
        "Inputs to ckks.multiply_plain must be of type ckks.ctxt & ckks.ptxt."); // Should never trigger
    assert(type_x.getElementType() == type_y.getElementType() && "Inputs to ckks.sub_plain must have same elementType.");
    inferredReturnTypes.push_back(CiphertextType::get(context, type_x.getSize(), type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::ExponentiateOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ExponentiateOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    assert(type_x && "First input to ckks.exponentiate must be of type !ckks.ctxt."); // Should never trigger
    auto new_size = type_x.getSize() + 1;
    inferredReturnTypes.push_back(CiphertextType::get(context, new_size, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::RelinearizeOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = RelinearizeOpAdaptor(operands, attributes, properties, regions);
    auto type_x = op.getX().getType().dyn_cast<CiphertextType>();
    assert(type_x && "First input to ckks.relinearize must be of type !ckks.ctxt."); // Should never trigger
    assert(type_x.getSize() == 3 && "Size of input to ckks.relinearize must be three!");
    inferredReturnTypes.push_back(CiphertextType::get(context, 2, type_x.getElementType()));
    return ::mlir::success();
}

::mlir::LogicalResult ckks::RotateOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = RotateOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    assert(type_x && "Input to ckks.rotate must be of type ckks.ctxt."); // Should never trigger
    inferredReturnTypes.push_back(type_x);
    return ::mlir::success();
}

::mlir::LogicalResult ckks::ModswitchOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ModswitchOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    assert(type_x && "Input to ckks.modswitch must be of type ckks.ctxt."); // Should never trigger
    inferredReturnTypes.push_back(type_x);
    return ::mlir::success();
}

::mlir::LogicalResult ckks::ModswitchPlainOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ModswitchPlainOpAdaptor(operands, attributes, properties, regions);
    PlaintextType type_x = op.getX().getType().dyn_cast<PlaintextType>();
    assert(type_x && "Input to ckks.modswitch_plain must be of type ckks.ptxt."); // Should never trigger
    inferredReturnTypes.push_back(type_x);
    return ::mlir::success();
}

::mlir::LogicalResult ckks::ModswitchToOp::inferReturnTypes(
    ::mlir::MLIRContext *context, ::std::optional<::mlir::Location> location, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::OpaqueProperties properties, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type> &inferredReturnTypes)
{
    auto op = ModswitchToOpAdaptor(operands, attributes, properties, regions);
    CiphertextType type_x = op.getX().getType().dyn_cast<CiphertextType>();
    CiphertextType type_y = op.getY().getType().dyn_cast<CiphertextType>();
    assert(type_x && type_y && "Inputs to ckks.modswitch_plain must be of type ckks.ctxt."); // Should never trigger
    inferredReturnTypes.push_back(type_y);
    return ::mlir::success();
}

/// simplifies a constant operation to its value (used for constant folding?)
::mlir::OpFoldResult ckks::ConstOp::fold(FoldAdaptor adaptor)
{
    return getValue();
}

/// simplifies away materialization(materialization(x)) to x if the types work
::mlir::OpFoldResult ckks::MaterializeOp::fold(FoldAdaptor adaptor)
{
    if (auto m_op = getInput().getDefiningOp<ckks::MaterializeOp>())
        if (m_op.getInput().getType() == getResult().getType())
            return m_op.getInput();
    return {};
}

/// simplifies away extract(v, 0) as scalars are simply "ctxt where we only care about slot 0"
::mlir::OpFoldResult ckks::ExtractOp::fold(FoldAdaptor adaptor)
{
    if (adaptor.getI().isZero())
        return getVector();
    else
        return nullptr;
}

//===----------------------------------------------------------------------===//
// CKKS dialect definitions
//===----------------------------------------------------------------------===//
#include "heco/IR/CKKS/CKKSDialect.cpp.inc"
void CKKSDialect::initialize()
{
    // Registers all the Types into the CKKSDialect class
    addTypes<
#define GET_TYPEDEF_LIST
#include "heco/IR/CKKS/CKKSTypes.cpp.inc"
        >();

    // Registers all the Operations into the CKKSDialect class
    addOperations<
#define GET_OP_LIST
#include "heco/IR/CKKS/CKKS.cpp.inc"
        >();
}