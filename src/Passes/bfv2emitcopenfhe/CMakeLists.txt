set(LLVM_TARGET_DEFINITIONS LowerBFVToEmitCOpenFHE.td)
mlir_tablegen(LowerBFVToEmitCOpenFHE.cpp.inc -gen-rewriters)
add_public_tablegen_target(bfv2emitcopenfhe)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerBFVToEmitCOpenFHE
        LowerBFVToEmitCOpenFHE.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        bfv2emitcopenfhe

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOBFVDialect
        MLIREmitCDialect
)

