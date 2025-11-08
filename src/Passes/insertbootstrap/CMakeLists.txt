add_heco_conversion_library(HECOInsertBootstrap
    InsertBootstrap.cpp

    #ADDITIONAL_HEADER_DIRS
    #Passes

    DEPENDS

    LINK_COMPONENTS
    Core

    LINK_LIBS PUBLIC
    HECOCKKSDialect
    MLIRFuncDialect
    MLIRPass
    MLIRTransforms
    )

