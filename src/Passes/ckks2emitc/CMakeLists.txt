set(LLVM_TARGET_DEFINITIONS LowerCKKSToEmitC.td)
mlir_tablegen(LowerCKKSToEmitC.cpp.inc -gen-rewriters)
add_public_tablegen_target(ckks2emitc)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerCKKSToEmitC
        LowerCKKSToEmitC.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        ckks2emitc

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOCKKSDialect
        MLIREmitCDialect
)

