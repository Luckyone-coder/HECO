#include "heco/Passes/insertbootstrap/InsertBootstrap.h"
#include "heco/IR/CKKS/CKKSDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include <cmath>
#include <iostream>

using namespace mlir;
using namespace heco;
using namespace ckks;

// ============================================================================
// 辅助函数实现
// ============================================================================

/// 判断类型是否为CKKS密文类型
static bool isCiphertextType(Type type) {
    return type.isa<ckks::CiphertextType>();
}

/// 获取操作消耗的乘法深度
static unsigned getOperationDepthCost(Operation *op) {
    // 加法、减法、旋转、重线性化不消耗深度
    if (isa<ckks::AddOp>(op) || isa<ckks::AddPlainOp>(op) ||
        isa<ckks::SubOp>(op) || isa<ckks::SubPlainOp>(op) ||
        isa<ckks::RelinearizeOp>(op) || isa<ckks::RotateOp>(op) ||
        isa<ckks::MultiplyPlainOp>(op) || 
        isa<ckks::AddManyOp>(op) ||
        isa<ckks::ModswitchOp>(op) || isa<ckks::ModswitchPlainOp>(op) ||
        isa<ckks::ModswitchToOp>(op)) {
        return 0;
    }
    
    // 密文乘法消耗1层深度
    if (isa<ckks::MultiplyOp>(op) || isa<ckks::SigmoidOp>(op)) {
        return 1;
    }
    
    // multiply_many消耗log2(n)的深度
    if (auto multiplyManyOp = dyn_cast<ckks::MultiplyManyOp>(op)) {
        unsigned numOperands = multiplyManyOp.getX().size();
        if (numOperands <= 1) return 0;
        return static_cast<unsigned>(std::ceil(std::log2(numOperands)));
    }
    
    // Bootstrap操作不消耗深度
    if (isa<ckks::BootstrapOp>(op)) {
        return 0;
    }
    
    // 默认不消耗深度
    return 0;
}

/// 判断是否需要对某个操作数进行Bootstrap
static bool needsBootstrap(Operation *op, Value operand, 
                          llvm::DenseMap<Value, unsigned> &depthMap,
                          unsigned minThreshold) {
    // 1. 检查操作数是否在深度映射表中
    if (!depthMap.count(operand)) {
        return false;  // 不是密文或未追踪，不需要Bootstrap
    }
    
    // 2. 不对明文类型进行Bootstrap
    if (!isCiphertextType(operand.getType())) {
        return false;
    }
    
    // 3. 获取操作数当前的可用深度
    unsigned currentDepth = depthMap[operand];
    
    // 4. 获取该操作将消耗的深度
    unsigned operationCost = getOperationDepthCost(op);
    
    // 5. 判断：如果执行后深度会低于阈值，需要Bootstrap
    // 公式：currentDepth < operationCost + minThreshold
    return (currentDepth < operationCost + minThreshold);
}

/// 更新操作执行后的深度信息
static void updateDepthAfterOperation(Operation *op, 
                                     llvm::DenseMap<Value, unsigned> &depthMap) {
    // 1. 找到所有密文输入中的最小深度
    unsigned minInputDepth = UINT_MAX;
    bool hasCiphertextInput = false;
    
    for (Value operand : op->getOperands()) {
        if (isCiphertextType(operand.getType()) && depthMap.count(operand)) {
            minInputDepth = std::min(minInputDepth, depthMap[operand]);
            hasCiphertextInput = true;
        }
    }
    
    // 2. 如果没有密文输入，无需更新
    if (!hasCiphertextInput || minInputDepth == UINT_MAX) {
        return;
    }
    
    // 3. 计算操作后的剩余深度
    unsigned depthCost = getOperationDepthCost(op);
    unsigned remainingDepth = (minInputDepth > depthCost) ? 
                              (minInputDepth - depthCost) : 0;
    
    // 4. 更新输出的深度
    for (Value result : op->getResults()) {
        if (isCiphertextType(result.getType())) {
            depthMap[result] = remainingDepth;
        }
    }
}

/// 为需要的操作数插入Bootstrap操作
static void insertBootstrapIfNeeded(Operation *op, IRRewriter &rewriter,
                                   llvm::DenseMap<Value, unsigned> &depthMap,
                                   unsigned freshDepth, unsigned minThreshold) {
    // 检查每个输入操作数
    for (OpOperand &operand : op->getOpOperands()) {
        Value val = operand.get();
        
        if (needsBootstrap(op, val, depthMap, minThreshold)) {
            // 在当前操作前插入bootstrap
            rewriter.setInsertionPoint(op);
            
            auto bootstrapOp = rewriter.create<ckks::BootstrapOp>(
                op->getLoc(),
                val.getType(),
                val
            );
            
            // 替换操作数
            operand.set(bootstrapOp.getOutput());
            
            // 更新深度映射
            depthMap[bootstrapOp.getOutput()] = freshDepth;
            
            // llvm::outs() << "[InsertBootstrap] Inserted bootstrap before operation: " 
            //              << op->getName() << " (depth was " << depthMap[val] 
            //              << ", restored to " << freshDepth << ")\n";
        }
    }
}

// ============================================================================
// Pass实现
// ============================================================================

void InsertBootstrapPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<ckks::CKKSDialect, func::FuncDialect>();
}

void InsertBootstrapPass::runOnOperation() {
    auto module = getOperation();
    IRRewriter rewriter(&getContext());
    
    // 深度映射表：Value -> 当前可用的乘法深度
    llvm::DenseMap<Value, unsigned> depthMap;
    
    // llvm::outs() << "[InsertBootstrap] Starting bootstrap insertion pass\n";
    // llvm::outs() << "  Initial depth: " << initialDepth << "\n";
    // llvm::outs() << "  Fresh depth: " << freshDepth << "\n";
    // llvm::outs() << "  Min threshold: " << minThreshold << "\n";
    
    // 遍历所有函数
    module.walk([&](func::FuncOp funcOp) {
        // llvm::outs() << "[InsertBootstrap] Processing function: " 
        //              << funcOp.getName() << "\n";
        
        // 1. 初始化输入参数的深度
        for (auto arg : funcOp.getArguments()) {
            if (isCiphertextType(arg.getType())) {
                depthMap[arg] = initialDepth;
                // llvm::outs() << "  Initialized arg depth: " << initialDepth << "\n";
            }
        }
        
        // 2. 收集所有CKKS操作（按执行顺序）
        llvm::SmallVector<Operation*, 64> ckksOps;
        funcOp.walk([&](Operation *op) {
            if (op->getDialect() && 
                op->getDialect()->getNamespace() == "ckks") {
                ckksOps.push_back(op);
            }
        });
        
        // 3. 按顺序处理每个CKKS操作
        for (Operation *op : ckksOps) {
            // 跳过Bootstrap操作本身
            if (isa<ckks::BootstrapOp>(op)) {
                continue;
            }
            
            // 检查并插入Bootstrap
            insertBootstrapIfNeeded(op, rewriter, depthMap, freshDepth, minThreshold);
            
            // 更新深度信息
            updateDepthAfterOperation(op, depthMap);
        }
    });
    
    // llvm::outs() << "[InsertBootstrap] Bootstrap insertion pass completed\n";
}

