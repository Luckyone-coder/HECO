set(LLVM_TARGET_DEFINITIONS LowerFHEToCKKS.td)
mlir_tablegen(LowerFHEToCKKS.cpp.inc -gen-rewriters)
add_public_tablegen_target(fhe2ckks)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

add_heco_conversion_library(HECOLowerFHEToCKKS
        LowerFHEToCKKS.cpp

        #ADDITIONAL_HEADER_DIRS
        #Passes

        DEPENDS
        fhe2ckks

        LINK_COMPONENTS
        Core

        LINK_LIBS PUBLIC
        HECOFHEDialect
        HECOCKKSDialect
)

