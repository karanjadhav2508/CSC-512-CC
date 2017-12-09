set(LLVM_LINK_COMPONENTS support)

add_clang_executable(t4
	FLT4.cpp
    )
    target_link_libraries(t4
      clangTooling
        clangBasic
	  clangASTMatchers
	    )
