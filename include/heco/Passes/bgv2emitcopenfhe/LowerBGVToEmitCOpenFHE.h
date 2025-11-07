#ifndef HECO_PASSES_BGV2EMITCOPENFHE_LOWERBGVTOEMITCOPENFHE_H_
#define HECO_PASSES_BGV2EMITCOPENFHE_LOWERBGVTOEMITCOPENFHE_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct LowerBGVToEmitCOpenFHEPass : public mlir::PassWrapper<LowerBGVToEmitCOpenFHEPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "bgv2emitcopenfhe";
    }
};

#endif
