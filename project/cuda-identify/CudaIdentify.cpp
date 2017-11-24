#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"
#include <string>
#include <iostream>
#include <iterator>

using namespace clang::tooling;
using namespace llvm;
using namespace clang;
using namespace clang::ast_matchers;
using namespace std;


// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");

int flag = 0;
Stmt *childBody;
std::vector<const FunctionDecl *> kernelFuncs;

clang::CUDAKernelCallExpr * hasCudaKernelCallExpr(Stmt *s) {
  Stmt * ckce;
  iterator_range<StmtIterator> s_children = s->children();
  for(StmtIterator child = s_children.begin(); child != s_children.end(); child++) {
    if(*child != NULL) {
      if(isa<CUDAKernelCallExpr>(*child)) {
	return (CUDAKernelCallExpr*) *child;
      }
      ckce = hasCudaKernelCallExpr(*child);
      if(ckce != NULL && isa<CUDAKernelCallExpr>(ckce)) {
	return (CUDAKernelCallExpr*) ckce;
      }
    }
  }
  return NULL;
}

std::string getCudaKernelCallExprName(Stmt *c) {
  for(StmtIterator child = c_children.begin(); child != c_children.end(); child++) {
    if(*child != NULL) {
      if(isa<DeclRefExpr>(*child)) {
        return ((DeclRefExpr*) *child)->getNameInfo().getName().getAsString();
      }
      ckceName = getCudaKernelCallExprName(*child);
      if(ckceName != "") {
        return ckceName;
      }
    }
  }
  return "";
}

class KernelFuncDefPrinter : public MatchFinder::MatchCallback {
public :
  virtual void run(const MatchFinder::MatchResult &Result) {
    const FunctionDecl *kf = Result.Nodes.getNodeAs<clang::FunctionDecl>("kernelFunc");
    CUDAKernelCallExpr *ck = hasCudaKernelCallExpr(kf->getBody());
    if(ck == NULL) {
      kernelFuncs.push_back(kf);
    }
    else {
      cout << "PARENT : " << kf->getNameInfo().getName().getAsString() << endl;
      std::string childKernelName = getCudaKernelCallExprName((Stmt*) ck);
      cout << "child kernel call : " << childKernelName << endl;
      for(std::vector<const FunctionDecl *>::iterator f = kernelFuncs.begin(); f != kernelFuncs.end(); f++) {
	if((*f)->getNameInfo().getName().getAsString() == childKernelName) {
          cout << "Found matching child kernel call : " << (*f)->getNameInfo().getName().getAsString() << endl;
	  break;
	}
      }
    }
  }
};

class MyASTConsumer: public ASTConsumer {
public:
  MyASTConsumer () {
    Finder.addMatcher(KernelFuncMatcher, &KernelFuncPrinter);
  }
  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    Finder.matchAST(Context);
  }

private:
  KernelFuncDefPrinter KernelFuncPrinter;
  //DeclarationMatcher KernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal), hasBody(compoundStmt(has(exprWithCleanups(has(cudaKernelCallExpr())))))).bind("parentKernelFunc");
  //DeclarationMatcher ParentKernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("parentKernelFunc");
//  DeclarationMatcher ParentKernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal), hasBody(compoundStmt(has(ifStmt(has(compoundStmt(has(ifStmt(has(compoundStmt(has(exprWithCleanups(has(cudaKernelCallExpr())))))))))))))).bind("parentKernelFunc");
  DeclarationMatcher KernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("kernelFunc");
  MatchFinder Finder;
};

// For each source file provided to the tool, a new FrontendAction is created.
class MyFrontendAction : public ASTFrontendAction {
public:
  MyFrontendAction() {}
  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &CI,
						 StringRef file) override {
    return llvm::make_unique<MyASTConsumer>();
  }
};


int main(int argc, const char **argv) {
  CommonOptionsParser OptionsParser(argc, argv, MyToolCategory);
  ClangTool Tool(OptionsParser.getCompilations(),
		 OptionsParser.getSourcePathList());

  return Tool.run(newFrontendActionFactory<MyFrontendAction>().get());
}
