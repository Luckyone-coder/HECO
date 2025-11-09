#include "heco/Passes/hir2hir/Tensor2BatchedSecret.h"
#include <iostream>
#include <memory>
#include "heco/IR/FHE/FHEDialect.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace heco;

void Tensor2BatchedSecretPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry
        .insert<fhe::FHEDialect, affine::AffineDialect, func::FuncDialect, scf::SCFDialect, tensor::TensorDialect>();
}

class ExtractPattern final : public OpConversionPattern<tensor::ExtractOp>
{
public:
    using OpConversionPattern<tensor::ExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        tensor::ExtractOp op, typename tensor::ExtractOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto tt = op.getTensor().getType().dyn_cast<TensorType>();
        int size = tt.hasStaticShape() ? tt.getNumElements() : -1;

        auto dstType = this->getTypeConverter()->convertType(op.getType());
        if (!dstType)
            return failure();
        if (auto st = dstType.dyn_cast_or_null<fhe::SecretType>())
        {
            auto batchedSecret = typeConverter->materializeTargetConversion(
                rewriter, op.getTensor().getLoc(),
                fhe::BatchedSecretType::get(getContext(), st.getPlaintextType(), size), op.getTensor());

            int64_t new_index = 0;
            int64_t current_offset = 0;
            for (int i = tt.getShape().size() - 1; i >= 0; --i)
            {
                // TODO: Support  tensor.extract indices format in fhe.extract,too
                auto cOp = op.getIndices()[i].getDefiningOp<arith::ConstantOp>();
                if (!cOp)
                {
                    emitError(
                        op.getLoc(), "tensor2fhe requires all tensor.extract indices used"
                                     " with tensors of fhe.secret to be constant!");
                    return failure();
                }
                auto index = cOp.getValueAttr().cast<IntegerAttr>().getValue().getLimitedValue();

                if (i == (int)tt.getShape().size() - 1)
                {
                    new_index = index;
                    current_offset = 1;
                }
                else // i < tt.getShape().size() - 1
                {
                    current_offset *= tt.getDimSize(i + 1);
                    new_index += index * current_offset;
                }
            }
            auto indexAttr = rewriter.getIndexAttr(new_index);
            rewriter.template replaceOpWithNewOp<fhe::ExtractOp>(op, dstType, batchedSecret, indexAttr);
        } // else do nothing
        return success();
    }
};

class InsertPattern final : public OpConversionPattern<tensor::InsertOp>
{
public:
    using OpConversionPattern<tensor::InsertOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        tensor::InsertOp op, typename tensor::InsertOpAdaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        auto dstType = this->getTypeConverter()->convertType(op.getType());
        if (!dstType)
            return failure();
        if (auto bst = dstType.dyn_cast_or_null<fhe::BatchedSecretType>())
        {
            auto batchedSecret =
                typeConverter->materializeTargetConversion(rewriter, op.getDest().getLoc(), bst, op.getDest());
            // TODO: Support  tensor.extract indices format in fhe.extract,too
            auto cOp = op.getIndices().front().getDefiningOp<arith::ConstantOp>();
            if (!cOp)
            {
                emitError(
                    op.getLoc(),
                    "tensor2fhe requires all tensor.insert indices used with tensors of fhe.secret to be constant!");
                return failure();
            }
            auto indexAttr = cOp.getValue().cast<IntegerAttr>();
            rewriter.template replaceOpWithNewOp<fhe::InsertOp>(op, dstType, op.getScalar(), batchedSecret, indexAttr);
        } // else do nothing
        return success();
    }
};

class ReturnPattern final : public OpConversionPattern<func::ReturnOp>
{
public:
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        if (op.getNumOperands() != 1)
        {
            emitError(op.getLoc(), "Currently only single value return operations are supported.");
            return failure();
        }
        auto dstType = this->getTypeConverter()->convertType(op.getOperandTypes().front());
        if (!dstType)
            return failure();
        if (auto bst = dstType.dyn_cast_or_null<fhe::BatchedSecretType>())
        {
            rewriter.setInsertionPoint(op);
            auto batchedSecret =
                typeConverter->materializeTargetConversion(rewriter, op.getLoc(), dstType, op.getOperands().front());
            rewriter.template replaceOpWithNewOp<func::ReturnOp>(op, batchedSecret);

        } // else do nothing
        return success();
    }
};

class FunctionPattern final : public OpConversionPattern<func::FuncOp>
{
public:
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::FuncOp op, typename func::FuncOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // Compute the new signature of the function.
        TypeConverter::SignatureConversion signatureConversion(op.getFunctionType().getNumInputs());
        SmallVector<Type> newResultTypes;
        if (failed(typeConverter->convertTypes(op.getFunctionType().getResults(), newResultTypes)))
            return failure();
        if (typeConverter->convertSignatureArgs(op.getFunctionType().getInputs(), signatureConversion).failed())
            return failure();
        auto new_functype = FunctionType::get(getContext(), signatureConversion.getConvertedTypes(), newResultTypes);

        rewriter.startRootUpdate(op);
        op.setType(new_functype);
        for (auto it = op.getRegion().args_begin(); it != op.getRegion().args_end(); ++it)
        {
            auto arg = *it;
            auto oldType = arg.getType();
            auto newType = typeConverter->convertType(oldType);
            arg.setType(newType);
            if (newType != oldType)
            {
                rewriter.setInsertionPointToStart(&op.getBody().getBlocks().front());
                auto m_op = typeConverter->materializeSourceConversion(rewriter, arg.getLoc(), oldType, arg);
                arg.replaceAllUsesExcept(m_op, m_op.getDefiningOp());
            }
        }
        rewriter.finalizeRootUpdate(op);

        return success();
    }
};

//修改

template <typename FHEOpTy>
class FHEGenericRewritePattern : public OpConversionPattern<FHEOpTy> {
public:
  using OpConversionPattern<FHEOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(FHEOpTy op, typename FHEOpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Convert result types
    SmallVector<Type, 4> newResultTypes;
    for (Type t : op->getResultTypes()) {
      Type ct = this->getTypeConverter()->convertType(t);
      if (!ct)
        return failure();
      newResultTypes.push_back(ct);
    }

    // New operands are the adaptor operands (the framework will materialize arguments if needed)
    SmallVector<Value, 8> newOperands(adaptor.getOperands().begin(), adaptor.getOperands().end());

    // Forward attributes from the original op so things like kernel or ArrayAttr are preserved
    ArrayRef<NamedAttribute> attrs = op->getAttrs();

    // Replace with new op that uses converted result types and forwarded attrs/operands
    rewriter.replaceOpWithNewOp<FHEOpTy>(op, newResultTypes, newOperands, attrs);
    return success();
  }
};

template <typename FHEOpTy>
class FHEUnaryRewritePattern : public OpConversionPattern<FHEOpTy> {
public:
  using OpConversionPattern<FHEOpTy>::OpConversionPattern;

  LogicalResult matchAndRewrite(FHEOpTy op, typename FHEOpTy::Adaptor adaptor,
                                ConversionPatternRewriter &rewriter) const override {
    // Single result op: convert that one result type
    Type resType = this->getTypeConverter()->convertType(op->getResult(0).getType());
    if (!resType)
      return failure();

    SmallVector<Value, 4> newOperands(adaptor.getOperands().begin(), adaptor.getOperands().end());
    ArrayRef<NamedAttribute> attrs = op->getAttrs();

    rewriter.replaceOpWithNewOp<FHEOpTy>(op, resType, newOperands, attrs);
    return success();
  }
};

// ---------- End: generic FHE rewrite patterns ----------

void Tensor2BatchedSecretPass::runOnOperation()
{
    auto type_converter = TypeConverter();

    type_converter.addConversion([&](TensorType t) {
        int size = t.hasStaticShape() ? t.getNumElements() : -1;
        if (auto st = t.getElementType().dyn_cast_or_null<fhe::SecretType>()) {
            return std::optional<Type>(fhe::BatchedSecretType::get(&getContext(), st.getPlaintextType(), size));
        } else {
            return std::optional<Type>(t);
        }
    });

    type_converter.addConversion([](Type t) {
        if (t.isa<TensorType>())
            return std::optional<Type>(std::nullopt);
        else
            return std::optional<Type>(t);
    });

    type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<fhe::BatchedSecretType>()) {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (auto tt = old_type.dyn_cast_or_null<TensorType>())
                if (auto st = tt.getElementType().dyn_cast_or_null<fhe::SecretType>())
                    return std::optional<Value>(builder.create<fhe::MaterializeOp>(loc, bst, vs));
        }
        return std::optional<Value>(std::nullopt);
    });

    type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<fhe::BatchedSecretType>()) {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (auto tt = old_type.dyn_cast_or_null<TensorType>())
                if (auto st = tt.getElementType().dyn_cast_or_null<fhe::SecretType>())
                    return std::optional<Value>(builder.create<fhe::MaterializeOp>(loc, bst, vs));
        }
        return std::optional<Value>(std::nullopt);
    });

    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto tt = t.dyn_cast_or_null<TensorType>()) {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (auto bst = old_type.dyn_cast_or_null<fhe::BatchedSecretType>())
                if (tt.getElementType() == bst.getCorrespondingSecretType())
                    return std::optional<Value>(builder.create<fhe::MaterializeOp>(loc, tt, vs));
        }
        return std::optional<Value>(std::nullopt);
    });

    ConversionTarget target(getContext());

    // Keep whole FHE dialect legal but force specific ops to be rewritten by marking them illegal
    target.addLegalDialect<fhe::FHEDialect>();

    // Force conversion of these FHE ops
    target.addIllegalOp<fhe::ConvOp>();
    target.addIllegalOp<fhe::ReluOp>();
    target.addIllegalOp<fhe::PoolOp>();
    target.addIllegalOp<fhe::FlattenOp>();
    target.addIllegalOp<fhe::FcOp>();
    target.addIllegalOp<fhe::SoftmaxOp>();

    // Tensor dialect legal only when converter thinks op is legal
    target.addDynamicallyLegalDialect<tensor::TensorDialect>([&](Operation *op) { return type_converter.isLegal(op); });

    // Functions and returns become legal only when all their signature types are legal
    target.addDynamicallyLegalOp<func::FuncOp>([&](Operation *op) {
        auto fop = llvm::dyn_cast<func::FuncOp>(op);
        for (auto t : op->getOperandTypes())
            if (!type_converter.isLegal(t)) return false;
        for (auto t : op->getResultTypes())
            if (!type_converter.isLegal(t)) return false;
        for (auto t : fop.getFunctionType().getInputs())
            if (!type_converter.isLegal(t)) return false;
        for (auto t : fop.getFunctionType().getResults())
            if (!type_converter.isLegal(t)) return false;
        return true;
    });

    target.addDynamicallyLegalOp<func::ReturnOp>([&](Operation *op) { return type_converter.isLegal(op->getOperandTypes()); });

    // Helper: mark an op legal only when all operand/result types are legal
    auto allTypesLegal = [&](Operation *op) {
        for (Type t : op->getOperandTypes())
            if (!type_converter.isLegal(t)) return false;
        for (Type t : op->getResultTypes())
            if (!type_converter.isLegal(t)) return false;
        return true;
    };

    target.addDynamicallyLegalOp<fhe::ConvOp>(allTypesLegal);
    target.addDynamicallyLegalOp<fhe::ReluOp>(allTypesLegal);
    target.addDynamicallyLegalOp<fhe::PoolOp>(allTypesLegal);
    target.addDynamicallyLegalOp<fhe::FlattenOp>(allTypesLegal);
    target.addDynamicallyLegalOp<fhe::FcOp>(allTypesLegal);
    target.addDynamicallyLegalOp<fhe::SoftmaxOp>(allTypesLegal);

    mlir::RewritePatternSet patterns(&getContext());
    patterns.add<ExtractPattern, InsertPattern, ReturnPattern, FunctionPattern>(type_converter, patterns.getContext());

    // Register FHE rewrite patterns (requires corresponding pattern template declarations present)
    patterns.add<FHEGenericRewritePattern<fhe::ConvOp>>(type_converter, patterns.getContext());
    patterns.add<FHEUnaryRewritePattern<fhe::ReluOp>>(type_converter, patterns.getContext());
    patterns.add<FHEGenericRewritePattern<fhe::PoolOp>>(type_converter, patterns.getContext());
    patterns.add<FHEUnaryRewritePattern<fhe::FlattenOp>>(type_converter, patterns.getContext());
    patterns.add<FHEGenericRewritePattern<fhe::FcOp>>(type_converter, patterns.getContext());
    patterns.add<FHEUnaryRewritePattern<fhe::SoftmaxOp>>(type_converter, patterns.getContext());

        if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns)))) {
        signalPassFailure();
    }
}