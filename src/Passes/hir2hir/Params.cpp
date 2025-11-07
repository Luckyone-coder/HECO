#include "heco/Passes/hir2hir/Params.h"
#include <iostream>
#include <memory>
#include "heco/IR/FHE/FHEDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"


#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
// #include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"


using namespace mlir;
using namespace heco;

void createParamsPass::getDependentDialects(mlir::DialectRegistry &registry) const
{
    registry.insert<emitc::EmitCDialect, func::FuncDialect>();
}

// inline mlir::ModuleOp getModule() { return getOperation(); }

// 函数：根据乘法深度计算N和q参数
std::pair<int, int> calculateParameters(int multiplication_depth) {
    int N, q;
    
    switch(multiplication_depth) {
        case 1:
            N = 2048;
            q = 54;
            break;
        case 2:
            N = 4096;
            q = 109;
            break;
        case 3:
        case 4:
            N = 8192;
            q = 218;
            break;
        case 5:
        case 6:
        case 7:
        case 8:
            N = 16384;
            q = 438;
            break;
        case 9:
        case 10:
        case 11:
        case 12:
        case 13:
        case 14:
        case 15:
            N = 32768;
            q = 881;
            break;
        default:
            // 对于未定义的乘法深度，使用默认值或插值计算
            if (multiplication_depth < 1) {
                N = 8192;
                q = 218;
            } else if (multiplication_depth > 30) {
                // 对于超过30的深度，使用线性插值
                N = 32768;
                q = 881;
            } else {
                // 理论上不会到达这里，但为了安全起见
                N = 32768;
                q = 881;
            }
            break;
    }
    
    return std::make_pair(N, q);
}

void createParamsPass::runOnOperation()
{ 
    // mlir::ModuleOp module = getOperation();
    // mlir::OpBuilder builder(module.getBodyRegion());
    // // auto &builder = rewriter.getBuilder();
    // auto loc = module.getLoc();
    auto& context = getContext();
     context.printOpOnDiagnostic(false); // 禁用诊断时打印操作


    // auto i64Type = IntegerType::get(&context, 64); // i64类型
    // FunctionType functionType = FunctionType::get(
    //     &context,
    //     {i64Type},        // 输入参数类型列表
    //     {}                // 返回值类型列表（空，表示void）
    // );

    // StringAttr funcName = builder.getStringAttr("keygen"); // 函数名

    // func::FuncOp keygenFunc = func::FuncOp::create(
    //     loc,
    //     funcName.getValue(), // 函数名
    //     functionType         // 函数类型
    // );

    // Block &entryBlock = *keygenFunc.addEntryBlock();
    // builder.setInsertionPointToStart(&entryBlock);

    // // 创建所需的OpaqueType类型
    // auto inputType = mlir::emitc::OpaqueType::get(
    //     builder.getContext(), 
    //     "std::vector<seal::Ciphertext>"
    // );
    // auto outputType = mlir::emitc::OpaqueType::get(
    //     builder.getContext(), 
    //     "seal::Ciphertext"
    // );

    // // 创建操作数列表（假设x是表示%3的Value对象）
    // ValueRange operands{};

    // // 创建函数名属性
    // auto funcNameAttr = builder.getStringAttr("evaluator_add_many");
    // auto funcNameAttr2 = builder.getStringAttr("evaluator_add_many2");

    // // 创建 CallOp（修正参数顺序）
    // auto callOp = builder.create<emitc::CallOp>(
    //     loc,
    //     TypeRange{outputType},         // 结果类型
    //     funcNameAttr,                  // 函数名（StringAttr 类型）
    //     // "","",
    //     ArrayAttr(), ArrayAttr(),      // 函数参数（空）
    //     // builder.getArrayAttr({}),      // 模板参数（空）
    //     operands                       // 操作数
    // );
    // auto callOp2 = builder.create<emitc::CallOp>(
    //     loc,
    //     TypeRange{outputType},         // 结果类型
    //     funcNameAttr2,                  // 函数名（StringAttr 类型）
    //     // "","",
    //     ArrayAttr(), ArrayAttr(),      // 函数参数（空）
    //     // builder.getArrayAttr({}),      // 模板参数（空）
    //     operands                       // 操作数
    // );

    // 获取结果值（即%4）
    // Value result = callOp.getResult(0);
    auto func = getOperation();
    func.walk([&](arith::ConstantOp constantOp) {
        // 获取常量值
        Attribute attr = constantOp.getValue();
        
        // 处理整数常量
        if (auto intAttr = attr.dyn_cast<IntegerAttr>()) {
            int64_t value = intAttr.getValue().getSExtValue();
            int N = calculateParameters(value).first;
            llvm::outs() << "get level: "<<value<<"\n";
            llvm::outs() << "generate params set: "<<"\n\n";
            llvm::outs() << "seal::EncryptionParameters parms(seal::scheme_type::bfv); " <<  "\n";
            llvm::outs() << "parms.set_poly_modulus_degree("<< N <<");" <<   "\n";
            llvm::outs() <<"parms.set_coeff_modulus(seal::CoeffModulus::BFVDefault(" << N << "));"<<   "\n";
            llvm::outs() <<"parms.set_plain_modulus(seal::PlainModulus::Batching("<< N <<", 20));"<<   "\n";
            llvm::outs() <<"seal::SEALContext context(parms);"<<   "\n";
            llvm::outs() <<"seal::KeyGenerator keygen(context);"<<   "\n";
            llvm::outs() <<"seal::SecretKey secret_key = keygen.secret_key();"<<   "\n";
            llvm::outs() <<"seal::PublicKey public_key;"<<   "\n";
            llvm::outs() <<"keygen.create_public_key(public_key);"<<   "\n";
            llvm::outs() <<"relinkeys = std::make_unique<seal::RelinKeys>();"<<   "\n";
            llvm::outs() <<"keygen.create_relin_keys(*relinkeys);"<<   "\n";
            llvm::outs() <<"galoiskeys = std::make_unique<seal::GaloisKeys>();"<<   "\n";
            llvm::outs() <<"keygen.create_galois_keys(*galoiskeys);"<<   "\n";
            llvm::outs() <<"encoder = std::make_unique<seal::BatchEncoder>(context);"<<   "\n";
            llvm::outs() <<"encryptor = std::make_unique<seal::Encryptor>(context, public_key);"<<   "\n";
            llvm::outs() << "evaluator = std::make_unique<seal::Evaluator>(context);"<<   "\n";
            llvm::outs() << "decryptor = std::make_unique<seal::Decryptor>(context, secret_key);"<<   "\n";
            // llvm::outs() << "Found integer constant: " << value << "\n";
            
            // 在这里可以使用获取到的数值进行参数选择
            // 例如：根据value调整FHE参数
        }
    });
    // context.printOpOnDiagnostic(false);
    // getOperation()->erase();  // 删除当前模块

}