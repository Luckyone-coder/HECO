#include "heco/Passes/fhe2ckks/LowerFHEToCKKS.h"
#include <iostream>
#include "heco/IR/CKKS/CKKSDialect.h"  // 需要创建这个方言
#include "heco/IR/FHE/FHEDialect.h"
#include "heco/IR/Poly/PolyDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/Debug.h"

using namespace mlir;
using namespace heco;

void LowerFHEToCKKSPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<ckks::CKKSDialect>();  // 需要创建这个方言
    registry.insert<poly::PolyDialect>();
}

// 旋转操作转换模式
class CKKSRotatePattern final : public OpConversionPattern<fhe::RotateOp>
{
protected:
    using OpConversionPattern<fhe::RotateOp>::typeConverter;

public:
    using OpConversionPattern<fhe::RotateOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::RotateOp op, typename fhe::RotateOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // 具体化操作数
        Value o = op.getOperand();
        Value materialized_operand;
        auto operandDstType = typeConverter->convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType)
            materialized_operand =
                typeConverter->materializeArgumentConversion(rewriter, op.getLoc(), operandDstType, o);
        else
            materialized_operand = o;

        rewriter.setInsertionPoint(op);
        auto poly_type = dstType.dyn_cast<ckks::CiphertextType>().getElementType();
        // TODO: MATCH PARAMETERS PROPERLY OR GET ACTUAL KEY FROM SOMEWHERE
        // auto key_type = ckks::GaloisKeysType::get(rewriter.getContext(), 0, 0, 0, poly_type);
        // auto keys = rewriter.create<ckks::LoadGaloisKeysOp>(op.getLoc(), key_type, "foo.glk", "glk.parms");
        rewriter.replaceOpWithNewOp<ckks::RotateOp>(op, dstType, materialized_operand, op.getI());
        return success();
    };
};

// 组合操作转换模式
class CKKSCombinePattern final : public OpConversionPattern<fhe::CombineOp>
{
protected:
    using OpConversionPattern<fhe::CombineOp>::typeConverter;

public:
    using OpConversionPattern<fhe::CombineOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::CombineOp op, typename fhe::CombineOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // TODO: Handle combining for CKKS!
        return failure();
    };
};

// 常量操作转换模式
class CKKSConstPattern final : public OpConversionPattern<fhe::ConstOp>
{
protected:
    using OpConversionPattern<fhe::ConstOp>::typeConverter;

public:
    using OpConversionPattern<fhe::ConstOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::ConstOp op, typename fhe::ConstOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // TODO: Handle const ops for CKKS!
        return failure();
    };
};

// 物化操作转换模式
class CKKSMaterializePattern final : public OpConversionPattern<fhe::MaterializeOp>
{
protected:
    using OpConversionPattern<fhe::MaterializeOp>::typeConverter;

public:
    using OpConversionPattern<fhe::MaterializeOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::MaterializeOp op, typename fhe::MaterializeOp::Adaptor adaptor,
        ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // 具体化操作数
        llvm::SmallVector<Value> materialized_operands;
        auto o = op.getOperand();
        auto operandDstType = typeConverter->convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType)
        {
            auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            assert(new_operand && "Type Conversion must not fail");
            materialized_operands.push_back(new_operand);
        }
        else
        {
            materialized_operands.push_back(o);
        }

        rewriter.replaceOpWithNewOp<ckks::MaterializeOp>(op, TypeRange(dstType), materialized_operands);
        return success();
    }
};

// 插入操作转换模式
class CKKSInsertPattern final : public OpConversionPattern<fhe::InsertOp>
{
protected:
    using OpConversionPattern<fhe::InsertOp>::typeConverter;

public:
    using OpConversionPattern<fhe::InsertOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::InsertOp op, typename fhe::InsertOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // 具体化操作数
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op.getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                assert(new_operand && "Type Conversion must not fail");
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }

        rewriter.replaceOpWithNewOp<ckks::InsertOp>(
            op, TypeRange(dstType), materialized_operands[0], materialized_operands[1], op.getIAttr());
        return success();
    }
};

// 提取操作转换模式
class CKKSExtractPattern final : public OpConversionPattern<fhe::ExtractOp>
{
protected:
    using OpConversionPattern<fhe::ExtractOp>::typeConverter;

public:
    using OpConversionPattern<fhe::ExtractOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        fhe::ExtractOp op, typename fhe::ExtractOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // 具体化操作数
        llvm::SmallVector<Value> materialized_operands;
        auto o = op.getOperand();
        auto operandDstType = typeConverter->convertType(o.getType());
        if (!operandDstType)
            return failure();
        if (o.getType() != operandDstType)
        {
            auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
            assert(new_operand && "Type Conversion must not fail");
            materialized_operands.push_back(new_operand);
        }
        else
        {
            materialized_operands.push_back(o);
        }

        rewriter.replaceOpWithNewOp<ckks::ExtractOp>(op, TypeRange(dstType), materialized_operands[0], op.getIAttr());
        return success();
    }
};

/// CKKS基本操作转换模式（针对无属性的操作）
template <typename OpType>
class CKKSBasicPattern final : public OpConversionPattern<OpType>
{
protected:
    using OpConversionPattern<OpType>::typeConverter;

public:
    using OpConversionPattern<OpType>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        OpType op, typename OpType::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        rewriter.setInsertionPoint(op);

        auto dstType = typeConverter->convertType(op.getType());
        if (!dstType)
            return failure();

        // 具体化操作数
        llvm::SmallVector<Value> materialized_operands;
        for (Value o : op->getOperands())
        {
            auto operandDstType = typeConverter->convertType(o.getType());
            if (!operandDstType)
                return failure();
            if (o.getType() != operandDstType)
            {
                auto new_operand = typeConverter->materializeTargetConversion(rewriter, op.getLoc(), operandDstType, o);
                assert(new_operand && "Type Conversion must not fail");
                materialized_operands.push_back(new_operand);
            }
            else
            {
                materialized_operands.push_back(o);
            }
        }
        // TODO: Handle ctxt-ptxt versions for all operations!

        // 加法操作
        if (std::is_same<OpType, fhe::AddOp>())
        {
            if (op->getNumOperands() > 2)
                rewriter.replaceOpWithNewOp<ckks::AddManyOp>(op, TypeRange(dstType), materialized_operands);
            else
                rewriter.replaceOpWithNewOp<ckks::AddOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // 减法操作
        else if (std::is_same<OpType, fhe::SubOp>())
        {
            rewriter.replaceOpWithNewOp<ckks::SubOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Sigmoid
        else if (std::is_same<OpType, fhe::SigmoidOp>())
        {
            rewriter.replaceOpWithNewOp<ckks::SigmoidOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Conv
        else if (std::is_same<OpType, fhe::ConvOp>())
        {
            rewriter.replaceOpWithNewOp<ckks::ConvOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Relu
        else if (std::is_same<OpType, fhe::ReluOp>())
        {
            rewriter.replaceOpWithNewOp<ckks::ReluOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Pool
        else if (std::is_same<OpType, fhe::PoolOp>())
        {
            rewriter.replaceOpWithNewOp<ckks::PoolOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Flatten
        else if (std::is_same<OpType, fhe::FlattenOp>())
        {
            rewriter.replaceOpWithNewOp<ckks::FlattenOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Fc
        else if (std::is_same<OpType, fhe::FcOp>())
        {
            rewriter.replaceOpWithNewOp<ckks::FcOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // Softmax
        else if (std::is_same<OpType, fhe::SoftmaxOp>())
        {
            rewriter.replaceOpWithNewOp<ckks::SoftmaxOp>(op, TypeRange(dstType), materialized_operands);
            return success();
        }
        // 乘法操作
        else if (std::is_same<OpType, fhe::MultiplyOp>())
        {
            if (op->getNumOperands() > 2)
            {
                // TODO: Handle ptxt in multiplies with more than two arguments
                rewriter.setInsertionPoint(op);
                auto poly_type = op.getType().template dyn_cast<ckks::CiphertextType>().getElementType();
                // TODO: MATCH PARAMETERS PROPERLY OR GET ACTUAL KEY FROM SOMEWHERE
                // auto key_type = ckks::RelinKeysType::get(rewriter.getContext(), 0, 0, 0, poly_type);
                // auto keys = rewriter.create<ckks::LoadRelinKeysOp>(op.getLoc(), key_type, "foo.rlk", "rlk.parms");
                auto new_op =
                    rewriter.replaceOpWithNewOp<ckks::MultiplyManyOp>(op, TypeRange(dstType), materialized_operands);
                rewriter.setInsertionPointAfter(new_op);
                // @TODO:可能CKKS需要重新缩放以控制噪声增长
            }
            else
            {
                // TODO: wHandle ptxt in first position, too!
                if (materialized_operands.size() == 2 && materialized_operands[1].getType().isa<ckks::PlaintextType>())
                    rewriter.replaceOpWithNewOp<ckks::MultiplyPlainOp>(op, TypeRange(dstType), materialized_operands);
                else
                {
                    auto new_op = rewriter.replaceOpWithNewOp<ckks::MultiplyOp>(op, TypeRange(dstType), materialized_operands);
                    // @TODO:可能CKKS需要重新缩放以控制噪声增长
                }
            }
            return success();
        }

        return failure();
    };
};

/// 函数转换模式（样板代码，不依赖于特定方言）
class CKKSFunctionConversionPattern final : public OpConversionPattern<func::FuncOp>
{
public:
    using OpConversionPattern<func::FuncOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::FuncOp op, typename func::FuncOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        // 计算函数的新签名
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

/// 更多不依赖方言的样板代码
class CKKSReturnPattern final : public OpConversionPattern<func::ReturnOp>
{
public:
    using OpConversionPattern<func::ReturnOp>::OpConversionPattern;

    LogicalResult matchAndRewrite(
        func::ReturnOp op, typename func::ReturnOp::Adaptor adaptor, ConversionPatternRewriter &rewriter) const override
    {
        if (op->getNumOperands() != 1)
        {
            emitError(op->getLoc(), "Only single value returns support for now.");
            return failure();
        }
        auto dstType = this->getTypeConverter()->convertType(op->getOperandTypes().front());
        if (!dstType)
            return failure();

        rewriter.setInsertionPoint(op);
        auto materialized =
            typeConverter->materializeTargetConversion(rewriter, op.getLoc(), dstType, op.getOperands());
        // 构建新的返回操作
        rewriter.replaceOpWithNewOp<func::ReturnOp>(op, materialized);

        return success();
    }
};

void LowerFHEToCKKSPass::runOnOperation()
{
    // TODO: We still need to emit a pre-amble with an include statement
    //  this should refer to some "magic file" that also sets up keys/etc and our custom evaluator wrapper for now

    auto type_converter = TypeConverter();

    type_converter.addConversion([&](Type t) {
        if (t.isa<fhe::BatchedSecretType>())
            // TODO: How to find the correct type here? CKKS specific parameters
            return std::optional<Type>(ckks::CiphertextType::get(
                &getContext(), 1, poly::PolynomialType::get(&getContext(), 2, true, 17, "parms.txt")));
        else if (t.isa<fhe::SecretType>())
            // TODO: How to find the correct type here? CKKS specific parameters
            return std::optional<Type>(ckks::CiphertextType::get(
                &getContext(), 1, poly::PolynomialType::get(&getContext(), 2, true, 17, "parms.txt")));
        else if (t.isIntOrIndexOrFloat())
            // CKKS主要处理浮点数，但也可以处理整数
            return std::optional<Type>(ckks::PlaintextType::get(
                &getContext(), poly::PolynomialType::get(&getContext(), 2, true, 17, "parms.txt")));
        else
            return std::optional<Type>(t);
    });

    type_converter.addTargetMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<ckks::CiphertextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>() || old_type.dyn_cast_or_null<fhe::SecretType>())
            {
                return std::optional<Value>(builder.create<ckks::MaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<ckks::PlaintextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.isIntOrIndexOrFloat())
            {
                return std::optional<Value>(builder.create<ckks::EncodeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
    });

    type_converter.addArgumentMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto ot = t.dyn_cast_or_null<ckks::CiphertextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.dyn_cast_or_null<fhe::BatchedSecretType>() || old_type.dyn_cast_or_null<fhe::SecretType>())
            {
                return std::optional<Value>(builder.create<ckks::MaterializeOp>(loc, ot, vs));
            }
        }
        else if (auto ot = t.dyn_cast_or_null<ckks::PlaintextType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materalize single values");
            auto old_type = vs.front().getType();
            if (old_type.isIntOrIndexOrFloat())
            {
                return std::optional<Value>(builder.create<ckks::EncodeOp>(loc, ot, vs));
            }
        }
        return std::optional<Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
    });

    type_converter.addSourceMaterialization([&](OpBuilder &builder, Type t, ValueRange vs, Location loc) {
        if (auto bst = t.dyn_cast_or_null<fhe::BatchedSecretType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<ckks::CiphertextType>())
                return std::optional<Value>(builder.create<ckks::MaterializeOp>(loc, bst, vs));
        }
        else if (auto st = t.dyn_cast_or_null<fhe::SecretType>())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<ckks::CiphertextType>())
                return std::optional<Value>(builder.create<ckks::MaterializeOp>(loc, st, vs));
        }
        else if (t.isIntOrIndexOrFloat())
        {
            assert(!vs.empty() && ++vs.begin() == vs.end() && "currently can only materialize single values");
            auto old_type = vs.front().getType();
            if (auto ot = old_type.dyn_cast_or_null<ckks::PlaintextType>())
                return std::optional<Value>(builder.create<ckks::DecodeOp>(loc, t, vs));
        }
        return std::optional<Value>(std::nullopt); /* would instead like to signal NO other conversions can be tried */
    });

    ConversionTarget target(getContext());
    target.addIllegalDialect<fhe::FHEDialect>();
    target.addLegalDialect<ckks::CKKSDialect>();
    target.addLegalOp<ModuleOp>();
    target.addLegalOp<fhe::ConstOp>();
    target.addLegalOp<fhe::CombineOp>();
    target.addDynamicallyLegalOp<func::FuncOp>([&](Operation *op) {
        auto fop = llvm::dyn_cast<func::FuncOp>(op);
        for (auto t : op->getOperandTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : op->getResultTypes())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getInputs())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        for (auto t : fop.getFunctionType().getResults())
        {
            if (!type_converter.isLegal(t))
                return false;
        }
        return true;
    });
    target.addDynamicallyLegalOp<func::ReturnOp>(
        [&](Operation *op) { return type_converter.isLegal(op->getOperandTypes()); });
    mlir::RewritePatternSet patterns(&getContext());

    patterns.add<
        CKKSFunctionConversionPattern, CKKSReturnPattern, CKKSBasicPattern<fhe::AddOp>,
        CKKSBasicPattern<fhe::SubOp>, CKKSBasicPattern<fhe::MultiplyOp>, CKKSRotatePattern, CKKSConstPattern,CKKSBasicPattern<fhe::SigmoidOp>, CKKSConstPattern,CKKSBasicPattern<fhe::ConvOp>,
        CKKSBasicPattern<fhe::ReluOp>, CKKSBasicPattern<fhe::PoolOp>, CKKSBasicPattern<fhe::FlattenOp>, CKKSBasicPattern<fhe::FcOp>, CKKSBasicPattern<fhe::SoftmaxOp>,
        CKKSMaterializePattern, CKKSInsertPattern, CKKSExtractPattern>(type_converter, patterns.getContext());

    if (mlir::failed(mlir::applyPartialConversion(getOperation(), target, std::move(patterns))))
        signalPassFailure();
}