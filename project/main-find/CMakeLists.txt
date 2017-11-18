set(LLVM_LINK_COMPONENTS support)

add_clang_executable(main-find
	MainFind.cpp
    )
    target_link_libraries(main-find
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
