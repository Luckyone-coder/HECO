//
// Created by rainyz on 25-7-7.
//

#ifndef LOWERCKKSTOOPENFHE_H
#define LOWERCKKSTOOPENFHE_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

struct LowerCKKSToEmitCOpenFHEPass : public mlir::PassWrapper<LowerCKKSToEmitCOpenFHEPass, mlir::OperationPass<mlir::ModuleOp>>
{
    void getDependentDialects(mlir::DialectRegistry &registry) const override;

    void runOnOperation() override;

    mlir::StringRef getArgument() const final
    {
        return "ckks2emitcopenfhe";
    }
};


#endif //LOWERCKKSTOOPENFHE_H
