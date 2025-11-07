set(LLVM_TARGET_DEFINITIONS LowerCKKSToEmitCOpenFHE.td)
mlir_tablegen(LowerCKKSToEmitcOpenFHE.cpp.inc -gen-rewriters)
add_public_tablegen_target(ckks2emitcopenfhe)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerCKKSToEmitCOpenFHE
        LowerCKKSToEmitCOpenFHE.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        ckks2emitcopenfhe

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOCKKSDialect
        MLIREmitCDialect
)

