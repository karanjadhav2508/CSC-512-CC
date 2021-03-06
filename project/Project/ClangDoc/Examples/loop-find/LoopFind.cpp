#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
// Declares clang::SyntaxOnlyAction.
#include "clang/Frontend/FrontendActions.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
// Declares llvm::cl::extrahelp.
#include "llvm/Support/CommandLine.h"

using namespace clang::tooling;
using namespace llvm;
using namespace clang;
using namespace clang::ast_matchers;


// Apply a custom category to all command-line options so that they are the
// only ones displayed.
static llvm::cl::OptionCategory MyToolCategory("my-tool options");

// CommonOptionsParser declares HelpMessage with a description of the common
// command-line options related to the compilation database and input files.
// It's nice to have this help message in all tools.
static cl::extrahelp CommonHelp(CommonOptionsParser::HelpMessage);

// A help message for this specific tool can be added afterwards.
static cl::extrahelp MoreHelp("\nMore help text...");

	  
class ForLoopPrinter : public MatchFinder::MatchCallback {
public :
  virtual void run(const MatchFinder::MatchResult &Result) {
    printf("------- A For Loop ----------\n");
    if (const ForStmt *FS = Result.Nodes.getNodeAs<clang::ForStmt>("forLoop"))
      FS->dump();
  }
};

class ForLoopSIPrinter : public MatchFinder::MatchCallback {
public :
  virtual void run(const MatchFinder::MatchResult &Result) {
    printf("------- A For Loop w/ Single Init----------\n");
    if (const ForStmt *FS = Result.Nodes.getNodeAs<clang::ForStmt>("forLoopSI"))
      FS->dump();
  }
};

class ForLoopCallPrinter : public MatchFinder::MatchCallback {
public :
  virtual void run(const MatchFinder::MatchResult &Result) {
    printf("------- A For Loop w/ Calls----------\n");
    if (const ForStmt *FS = Result.Nodes.getNodeAs<clang::ForStmt>("callInLoop"))
      FS->dump();
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
    Finder.addMatcher(ForLoopMatcher, &ForPrinter);
    Finder.addMatcher(ForLoopMatcher_withSI, &ForSIPrinter);
    Finder.addMatcher(ForLoopMatcher_withCall, &ForCallPrinter);
  }
  void HandleTranslationUnit(ASTContext &Context) override {
    // Run the matchers when we have the whole TU parsed.
    Finder.matchAST(Context);
  }

private:
  ForLoopPrinter ForPrinter;
  ForLoopSIPrinter ForSIPrinter;
  ForLoopCallPrinter ForCallPrinter;
  StatementMatcher ForLoopMatcher= forStmt(hasLoopInit(declStmt(hasSingleDecl(varDecl(hasName("i")))))).bind("forLoop");
  StatementMatcher ForLoopMatcher_withSI=
      forStmt(hasLoopInit(declStmt(hasSingleDecl(varDecl(hasInitializer(integerLiteral(equals(0)))))))).bind("forLoopSI");
  StatementMatcher ForLoopMatcher_withCall=
      forStmt(hasBody(compoundStmt(has(callExpr())))).bind("callInLoop");
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
