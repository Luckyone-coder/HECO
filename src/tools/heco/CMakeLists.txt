get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
message(  ${dialect_libs} )
set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIREmitCDialect
        MLIROptLib
        HECOFHEDialect
        HECOEVADialect
        HECOBFVDialect
        HECOCKKSDialect
        HECOBGVDialect
        HECOPolyDialect
        MLIRAnalysis
        MLIRCallInterfaces
        MLIRCastInterfaces
        MLIRExecutionEngine
        MLIRIR
        MLIRLLVMCommonConversion
        MLIRLLVMToLLVMIRTranslation
        MLIRMemRefDialect
        MLIRParser
        MLIRPass
        MLIRInferTypeOpInterface
        MLIRControlFlowInterfaces
        MLIRSideEffectInterfaces
        MLIRTargetLLVMIRExport
        MLIRTransforms  
        HECOLowerBFVToEmitC
        HECOLowerCKKSToEmitC
        HECOLowerBGVToEmitC
        HECOLowerBFVToLLVM
        HECOLowerFHEToBFV
        HECOLowerFHEToCKKS
        HECOLowerFHEToBGV
        HECOLowerCKKSToEmitCOpenFHE
        HECOLowerBFVToEmitCOpenFHE
        HECOLowerBGVToEmitCOpenFHE
        HECOLowerFHEToEmitC
        HECOLowerFHEToEVA
        HECOInsertRelinearize
        HECOWaterlineRescale
        HECOMatchScale
        HECOMarkMetadata
        HECOLazyModswitch
        HECOCatchAll
        )
add_llvm_executable(heco heco.cpp)

llvm_update_compile_flags(heco)
target_link_libraries(heco PRIVATE ${LIBS})

add_dependencies(heco bfv2emitc)
add_dependencies(heco ckks2emitc)
add_dependencies(heco bgv2emitc)
add_dependencies(heco ckks2emitcopenfhe)
add_dependencies(heco bfv2emitcopenfhe)
add_dependencies(heco bgv2emitcopenfhe)

mlir_check_all_link_libraries(heco)
