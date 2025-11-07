#ifndef LOWERCKKSTOEMITC_H
#define LOWERCKKSTOEMITC_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct LowerCKKSToEmitCPass : public mlir::PassWrapper<LowerCKKSToEmitCPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "ckks2emitc";
    }
};

#endif //LOWERCKKSTOEMITC_H
