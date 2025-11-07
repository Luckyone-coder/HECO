#ifndef HECO_PASSES_HIR2HIR_createParams_H_
#define HECO_PASSES_HIR2HIR_createParams_H_

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


struct createParamsPass : public mlir::PassWrapper<createParamsPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "createParams";
    }
};

#endif // HECO_PASSES_HIR2HIR_createParams_H_
