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
std::vector<clang::FunctionDecl *> kernelFuncs;

class ParentKernelFuncDefPrinter : public MatchFinder::MatchCallback {
public :
  virtual void run(const MatchFinder::MatchResult &Result) {
    const FunctionDecl *m = Result.Nodes.getNodeAs<clang::FunctionDecl>("parentKernelFunc");
    cout << "PARENT : " << m->getNameInfo().getName().getAsString() << endl;
    /*
    std::string func = m->getNameInfo().getName().getAsString();
    if(flag==0) {
	    cout << "ChildFunc : " << func << endl;
	    childBody = m->getBody();
	    flag = 1;
	    //childBody->dumpColor();
    }
    else {
	    cout << "ParentFunc : " << func << endl;
	    //m->dumpColor();
	    cout << "X------------------------------------------------------------------------X" << endl;
	    llvm::iterator_range<clang::StmtIterator> parentFuncStmts = m->getBody()->children();
	    StmtIterator s;
            for(s = parentFuncStmts.begin(); s != parentFuncStmts.end(); s++) {
	      if(isa<ExprWithCleanups>(*s)) {
		StmtIterator ewcIter;
	        llvm::iterator_range<clang::StmtIterator> ewcChildren = s->children();
		for(ewcIter = ewcChildren.begin(); ewcIter != ewcChildren.end(); ewcIter++) {
		  if(isa<CUDAKernelCallExpr>(*ewcIter)) {
		    StmtIterator ckIter;
		    llvm::iterator_range<clang::StmtIterator> ckChildren = ewcIter->children();
		    for(ckIter = ckChildren.begin(); ckIter != ckChildren.end(); ckIter++) {
		      if(isa<CallExpr>(*ckIter)) {
			StmtIterator ceIter;
			llvm::iterator_range<clang::StmtIterator> ceChildren = ckIter->children();
			for(ceIter = ceChildren.begin(); ceIter != ceChildren.end(); ceIter++) {
			  if(isa<CXXConstructExpr>(*ceIter)) {
			    //straight to IntegerLiteral from CXXConstructExpr
			    //ceIter->child_begin()->child_begin()->child_begin()->child_begin()->child_begin()->child_begin()->dumpColor();
			  }
			}
		      }
		    }
		  }
		}
	      }
	    }
    }
    */
  }
};

class KernelFuncDefPrinter : public MatchFinder::MatchCallback {
public :
  virtual void run(const MatchFinder::MatchResult &Result) {
    const FunctionDecl *m = Result.Nodes.getNodeAs<clang::FunctionDecl>("kernelFunc");
    //kernelFuncs.push_back(m);
    cout << "KERNEL : " << m->getNameInfo().getName().getAsString() << endl;
  }
};

class MyASTConsumer: public ASTConsumer {
public:
  MyASTConsumer () {
    /*    ForLoopMatcher = forStmt().bind("forLoop");
    ForLoopMatcher_withSI =
      forStmt(hasLoopInit(declStmt(hasSingleDecl(varDecl(hasInitializer(integerLiteral(equals(0)))))))).bind("forLoopSI");
    ForLoopMatcher_withCall =
      forStmt(hasBody(compoundStmt(has(callExpr())))).bind("callInLoop");
    */
    //Finder.addMatcher(ForLoopMatcher, &ForPrinter);
    //Finder.addMatcher(ForLoopMatcher_withSI, &ForSIPrinter);
    //Finder.addMatcher(ForLoopMatcher_withCall, &ForCallPrinter);
    Finder.addMatcher(ParentKernelFuncMatcher, &ParentKernelFuncPrinter);
    Finder.addMatcher(KernelFuncMatcher, &KernelFuncPrinter);
  }
  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    Finder.matchAST(Context);
  }

private:
  ParentKernelFuncDefPrinter ParentKernelFuncPrinter;
  KernelFuncDefPrinter KernelFuncPrinter;
  //DeclarationMatcher KernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal), hasBody(compoundStmt(has(exprWithCleanups(has(cudaKernelCallExpr())))))).bind("parentKernelFunc");
  //DeclarationMatcher KernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal)).bind("parentKernelFunc");
  DeclarationMatcher ParentKernelFuncMatcher = functionDecl(hasAttr(clang::attr::CUDAGlobal), hasBody(compoundStmt(has(ifStmt(has(compoundStmt(has(ifStmt(has(compoundStmt(has(exprWithCleanups(has(cudaKernelCallExpr())))))))))))))).bind("parentKernelFunc");
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
