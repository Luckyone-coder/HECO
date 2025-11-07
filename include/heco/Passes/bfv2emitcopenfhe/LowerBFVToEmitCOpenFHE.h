#ifndef HECO_PASSES_BFV2EMITCOPENFHE_LOWERBFVTOEMITCOPENFHE_H_
#define HECO_PASSES_BFV2EMITCOPENFHE_LOWERBFVTOEMITCOPENFHE_H_

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct LowerBFVToEmitCOpenFHEPass : public mlir::PassWrapper<LowerBFVToEmitCOpenFHEPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "bfv2emitcopenfhe";
    }
};

#endif
