#ifndef HECO_PASSES_INSERTBOOTSTRAP_INSERTBOOTSTRAP_H_
#define HECO_PASSES_INSERTBOOTSTRAP_INSERTBOOTSTRAP_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace heco {

/// Pass to insert bootstrap operations when CKKS multiplicative depth is insufficient
struct InsertBootstrapPass : public mlir::PassWrapper<InsertBootstrapPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;
    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "insert-ckks-bootstrap";
    }

    mlir::StringRef getDescription() const final
    {
        return "Insert bootstrap operations when CKKS multiplicative depth reaches threshold";
    }

    // 可配置参数
    unsigned initialDepth = 3;     // 输入密文的初始乘法深度
    unsigned freshDepth = 3;       // Bootstrap后的新深度
    unsigned minThreshold = 1;      // 触发Bootstrap的最小深度阈值
};

} // namespace heco

#endif // HECO_PASSES_INSERTBOOTSTRAP_INSERTBOOTSTRAP_H_

