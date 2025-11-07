set(LLVM_TARGET_DEFINITIONS LowerBGVToEmitCOpenFHE.td)
mlir_tablegen(LowerBGVToEmitCOpenFHE.cpp.inc -gen-rewriters)
add_public_tablegen_target(bgv2emitcopenfhe)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerBGVToEmitCOpenFHE
        LowerBGVToEmitCOpenFHE.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        bgv2emitcopenfhe

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOBGVDialect
        MLIREmitCDialect
)

