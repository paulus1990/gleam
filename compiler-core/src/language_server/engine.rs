use crate::{
    ast::{
        self, Arg, Definition, Function, Import, ModuleConstant, SrcSpan, TypedDefinition,
        TypedExpr, TypedPattern, Statement, Assignment, CallArg,
    },
    build::{Located, Module},
    config::PackageConfig,
    io::{CommandExecutor, FileSystemReader, FileSystemWriter},
    language_server::{
        compiler::LspProjectCompiler, files::FileSystemProxy, progress::ProgressReporter,
    },
    line_numbers::LineNumbers,
    paths::ProjectPaths,
    type_::{pretty::Printer, PreludeType, Type, ValueConstructorVariant},
    Error, Result, Warning,
};
use camino::Utf8PathBuf;
use ecow::EcoString;
use lsp::CodeAction;
use lsp_types::{self as lsp, Hover, HoverContents, MarkedString, Url};
use spdx::Expression;
use std::{sync::Arc, ops::Deref, cell::RefMut};
use strum::IntoEnumIterator;

use super::{
    code_action::CodeActionBuilder, src_span_to_lsp_range, DownloadDependencies, MakeLocker,
};

#[derive(Debug, PartialEq, Eq)]
pub struct Response<T> {
    pub result: Result<T, Error>,
    pub warnings: Vec<Warning>,
    pub compilation: Compilation,
}

#[derive(Debug, PartialEq, Eq)]
pub enum Compilation {
    /// Compilation was attempted and succeeded for these modules.
    Yes(Vec<Utf8PathBuf>),
    /// Compilation was not attempted for this operation.
    No,
}

#[derive(Debug)]
pub struct LanguageServerEngine<IO, Reporter> {
    pub(crate) paths: ProjectPaths,

    /// A compiler for the project that supports repeat compilation of the root
    /// package.
    /// In the event the the project config changes this will need to be
    /// discarded and reloaded to handle any changes to dependencies.
    pub(crate) compiler: LspProjectCompiler<FileSystemProxy<IO>>,

    modules_compiled_since_last_feedback: Vec<Utf8PathBuf>,
    compiled_since_last_feedback: bool,

    // Used to publish progress notifications to the client without waiting for
    // the usual request-response loop.
    progress_reporter: Reporter,
}

impl<'a, IO, Reporter> LanguageServerEngine<IO, Reporter>
where
    // IO to be supplied from outside of gleam-core
    IO: FileSystemReader
        + FileSystemWriter
        + CommandExecutor
        + DownloadDependencies
        + MakeLocker
        + Clone,
    // IO to be supplied from inside of gleam-core
    Reporter: ProgressReporter + Clone + 'a,
{
    pub fn new(
        config: PackageConfig,
        progress_reporter: Reporter,
        io: FileSystemProxy<IO>,
        paths: ProjectPaths,
    ) -> Result<Self> {
        let locker = io.inner().make_locker(&paths, config.target)?;

        // Download dependencies to ensure they are up-to-date for this new
        // configuration and new instance of the compiler
        progress_reporter.dependency_downloading_started();
        let manifest = io.inner().download_dependencies(&paths);
        progress_reporter.dependency_downloading_finished();

        // NOTE: This must come after the progress reporter has finished!
        let manifest = manifest?;

        let compiler =
            LspProjectCompiler::new(manifest, config, paths.clone(), io.clone(), locker)?;

        Ok(Self {
            modules_compiled_since_last_feedback: vec![],
            compiled_since_last_feedback: false,
            progress_reporter,
            compiler,
            paths,
        })
    }

    pub fn compile_please(&mut self) -> Response<()> {
        self.respond(Self::compile)
    }

    /// Compile the project if we are in one. Otherwise do nothing.
    fn compile(&mut self) -> Result<(), Error> {
        self.compiled_since_last_feedback = true;

        self.progress_reporter.compilation_started();
        let result = self.compiler.compile();
        self.progress_reporter.compilation_finished();

        let modules = result?;
        self.modules_compiled_since_last_feedback.extend(modules);

        Ok(())
    }

    fn take_warnings(&mut self) -> Vec<Warning> {
        self.compiler.take_warnings()
    }

    // TODO: test local variables
    // TODO: test same module constants
    // TODO: test imported module constants
    // TODO: test unqualified imported module constants
    // TODO: test same module records
    // TODO: test imported module records
    // TODO: test unqualified imported module records
    // TODO: test same module functions
    // TODO: test module function calls
    // TODO: test different package module function calls
    //
    //
    //
    // TODO: implement unqualified imported module functions
    // TODO: implement goto definition of modules that do not belong to the top
    // level package.
    //
    pub fn goto_definition(
        &mut self,
        params: lsp::GotoDefinitionParams,
    ) -> Response<Option<lsp::Location>> {
        self.respond(|this| {
            let params = params.text_document_position_params;
            let (line_numbers, node) = match this.node_at_position(&params) {
                Some(location) => location,
                None => return Ok(None),
            };

            let location = match node.definition_location() {
                Some(location) => location,
                None => return Ok(None),
            };

            let (uri, line_numbers) = match location.module {
                None => (params.text_document.uri, &line_numbers),
                Some(name) => {
                    let module = match this.compiler.get_source(name) {
                        Some(module) => module,
                        // TODO: support goto definition for functions defined in
                        // different packages. Currently it is not possible as the
                        // required LineNumbers and source file path information is
                        // not stored in the module metadata.
                        None => return Ok(None),
                    };
                    let url = Url::parse(&format!("file:///{}", &module.path))
                        .expect("goto definition URL parse");
                    (url, &module.line_numbers)
                }
            };
            let range = src_span_to_lsp_range(location.span, line_numbers);

            Ok(Some(lsp::Location { uri, range }))
        })
    }

    pub fn completion(
        &mut self,
        params: lsp::TextDocumentPositionParams,
    ) -> Response<Option<Vec<lsp::CompletionItem>>> {
        self.respond(|this| {
            let module = match this.module_for_uri(&params.text_document.uri) {
                Some(m) => m,
                None => return Ok(None),
            };

            let line_numbers = LineNumbers::new(&module.code);
            let byte_index =
                line_numbers.byte_index(params.position.line, params.position.character);

            let Some(found) = module.find_node(byte_index) else {
                return Ok(None);
            };

            let completions = match found {
                Located::Pattern(_pattern) => None,

                Located::Statement(_) | Located::Expression(_) => {
                    Some(this.completion_values(module))
                }

                Located::ModuleStatement(Definition::Function(_)) => {
                    Some(this.completion_types(module))
                }

                Located::FunctionBody(_) => Some(this.completion_values(module)),

                Located::ModuleStatement(Definition::TypeAlias(_) | Definition::CustomType(_)) => {
                    Some(this.completion_types(module))
                }

                Located::ModuleStatement(Definition::Import(_) | Definition::ModuleConstant(_)) => {
                    None
                }

                Located::Arg(_) => None,
            };

            Ok(completions)
        })
    }

    pub fn action(&mut self, params: lsp::CodeActionParams) -> Response<Option<Vec<CodeAction>>> {
        self.respond(|this| {
            let mut actions = vec![];
            let Some(module) = this.module_for_uri(&params.text_document.uri) else {
                return Ok(None);
            };

            code_action_unused_imports(module, &params, &mut actions);
            gleam_pipeline_suggestions(module, &params, &mut actions);

            Ok(if actions.is_empty() {
                None
            } else {
                Some(actions)
            })
        })
    }

    fn respond<T>(&mut self, handler: impl FnOnce(&mut Self) -> Result<T>) -> Response<T> {
        let result = handler(self);
        let warnings = self.take_warnings();
        // TODO: test. Ensure hover doesn't report as compiled
        let compilation = if self.compiled_since_last_feedback {
            let modules = std::mem::take(&mut self.modules_compiled_since_last_feedback);
            self.compiled_since_last_feedback = false;
            Compilation::Yes(modules)
        } else {
            Compilation::No
        };
        Response {
            result,
            warnings,
            compilation,
        }
    }

    pub fn hover(&mut self, params: lsp::HoverParams) -> Response<Option<Hover>> {
        self.respond(|this| {
            let params = params.text_document_position_params;

            let (lines, found) = match this.node_at_position(&params) {
                Some(value) => value,
                None => return Ok(None),
            };

            Ok(match found {
                Located::Statement(_) => None, // TODO: hover for statement
                Located::ModuleStatement(Definition::Function(fun)) => {
                    Some(hover_for_function_head(fun, lines))
                }
                Located::ModuleStatement(Definition::ModuleConstant(constant)) => {
                    Some(hover_for_module_constant(constant, lines))
                }
                Located::ModuleStatement(_) => None,
                Located::Pattern(pattern) => Some(hover_for_pattern(pattern, lines)),
                Located::Expression(expression) => Some(hover_for_expression(expression, lines)),
                Located::Arg(arg) => Some(hover_for_function_argument(arg, lines)),
                Located::FunctionBody(_) => None,
            })
        })
    }

    fn module_node_at_position(
        &self,
        params: &lsp::TextDocumentPositionParams,
        module: &'a Module,
    ) -> Option<(LineNumbers, Located<'a>)> {
        let line_numbers = LineNumbers::new(&module.code);
        let byte_index = line_numbers.byte_index(params.position.line, params.position.character);
        let node = module.find_node(byte_index);
        let node = node?;
        Some((line_numbers, node))
    }

    fn node_at_position(
        &self,
        params: &lsp::TextDocumentPositionParams,
    ) -> Option<(LineNumbers, Located<'_>)> {
        let module = self.module_for_uri(&params.text_document.uri)?;
        self.module_node_at_position(params, module)
    }

    fn module_for_uri(&self, uri: &Url) -> Option<&Module> {
        use itertools::Itertools;

        // The to_file_path method is available on these platforms
        #[cfg(any(unix, windows, target_os = "redox", target_os = "wasi"))]
        let path = uri.to_file_path().expect("URL file");

        #[cfg(not(any(unix, windows, target_os = "redox", target_os = "wasi")))]
        let path: Utf8PathBuf = uri.path().into();

        let components = path
            .strip_prefix(self.paths.root())
            .ok()?
            .components()
            .skip(1)
            .map(|c| c.as_os_str().to_string_lossy());
        let module_name: EcoString = Itertools::intersperse(components, "/".into())
            .collect::<String>()
            .strip_suffix(".gleam")?
            .into();

        self.compiler.modules.get(&module_name)
    }

    fn completion_types<'b>(&'b self, module: &'b Module) -> Vec<lsp::CompletionItem> {
        let mut completions = vec![];

        // Prelude types
        for type_ in PreludeType::iter() {
            completions.push(lsp::CompletionItem {
                label: type_.name().into(),
                detail: Some("Type".into()),
                kind: Some(lsp::CompletionItemKind::CLASS),
                ..Default::default()
            });
        }

        // Module types
        for (name, type_) in &module.ast.type_info.types {
            completions.push(type_completion(None, name, type_));
        }

        // Imported modules
        for import in module.ast.definitions.iter().filter_map(get_import) {
            // The module may not be known of yet if it has not previously
            // compiled yet in this editor session.
            // TODO: test getting completions from modules defined in other packages
            let Some(module) = self.compiler.get_module_inferface(&import.module) else {
                continue;
            };

            // Qualified types
            for (name, type_) in &module.types {
                if !type_.public {
                    continue;
                }

                let module = import.used_name();
                if module.is_some() {
                    completions.push(type_completion(module.as_ref(), name, type_));
                }
            }

            // Unqualified types
            for unqualified in &import.unqualified_values {
                let Some(type_) = module.get_public_type(&unqualified.name) else {
                    continue;
                };
                completions.push(type_completion(None, unqualified.used_name(), type_));
            }
        }

        completions
    }

    fn completion_values<'b>(&'b self, module: &'b Module) -> Vec<lsp::CompletionItem> {
        let mut completions = vec![];

        // Module functions
        for (name, value) in &module.ast.type_info.values {
            completions.push(value_completion(None, name, value));
        }

        // Imported modules
        for import in module.ast.definitions.iter().filter_map(get_import) {
            // The module may not be known of yet if it has not previously
            // compiled yet in this editor session.
            // TODO: test getting completions from modules defined in other packages
            let Some(module) = self.compiler.get_module_inferface(&import.module) else {
                continue;
            };

            // Qualified values
            for (name, value) in &module.values {
                if !value.public {
                    continue;
                }

                let module = import.used_name();
                if module.is_some() {
                    completions.push(value_completion(module.as_deref(), name, value));
                }
            }

            // Unqualified values
            for unqualified in &import.unqualified_values {
                let Some(value) = module.get_public_value(&unqualified.name) else {
                    continue;
                };
                completions.push(value_completion(None, unqualified.used_name(), value));
            }
        }

        completions
    }
}

fn type_completion(
    module: Option<&EcoString>,
    name: &str,
    type_: &crate::type_::TypeConstructor,
) -> lsp::CompletionItem {
    let label = match module {
        Some(module) => format!("{module}.{name}"),
        None => name.to_string(),
    };

    let kind = Some(if type_.typ.is_variable() {
        lsp::CompletionItemKind::VARIABLE
    } else {
        lsp::CompletionItemKind::CLASS
    });

    lsp::CompletionItem {
        label,
        kind,
        detail: Some("Type".into()),
        ..Default::default()
    }
}

fn value_completion(
    module: Option<&str>,
    name: &str,
    value: &crate::type_::ValueConstructor,
) -> lsp::CompletionItem {
    let label = match module {
        Some(module) => format!("{module}.{name}"),
        None => name.to_string(),
    };

    let type_ = Printer::new().pretty_print(&value.type_, 0);

    let kind = Some(match value.variant {
        ValueConstructorVariant::LocalVariable { .. } => lsp::CompletionItemKind::VARIABLE,
        ValueConstructorVariant::ModuleConstant { .. } => lsp::CompletionItemKind::CONSTANT,
        ValueConstructorVariant::LocalConstant { .. } => lsp::CompletionItemKind::CONSTANT,
        ValueConstructorVariant::ModuleFn { .. } => lsp::CompletionItemKind::FUNCTION,
        ValueConstructorVariant::Record { arity: 0, .. } => lsp::CompletionItemKind::ENUM_MEMBER,
        ValueConstructorVariant::Record { .. } => lsp::CompletionItemKind::CONSTRUCTOR,
    });

    let documentation = value.get_documentation().map(|d| {
        lsp::Documentation::MarkupContent(lsp::MarkupContent {
            kind: lsp::MarkupKind::Markdown,
            value: d.to_string(),
        })
    });

    lsp::CompletionItem {
        label,
        kind,
        detail: Some(type_),
        documentation,
        ..Default::default()
    }
}

fn get_import(statement: &TypedDefinition) -> Option<&Import<EcoString>> {
    match statement {
        Definition::Import(import) => Some(import),
        _ => None,
    }
}

fn hover_for_pattern(pattern: &TypedPattern, line_numbers: LineNumbers) -> Hover {
    let documentation = pattern.get_documentation().unwrap_or_default();

    // Show the type of the hovered node to the user
    let type_ = Printer::new().pretty_print(pattern.type_().as_ref(), 0);
    let contents = format!(
        "```gleam
{type_}
```
{documentation}"
    );
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(pattern.location(), &line_numbers)),
    }
}

fn hover_for_function_head(
    fun: &Function<Arc<Type>, TypedExpr>,
    line_numbers: LineNumbers,
) -> Hover {
    let empty_str = EcoString::from("");
    let documentation = fun.documentation.as_ref().unwrap_or(&empty_str);
    let function_type = Type::Fn {
        args: fun.arguments.iter().map(|arg| arg.type_.clone()).collect(),
        retrn: fun.return_type.clone(),
    };
    let formatted_type = Printer::new().pretty_print(&function_type, 0);
    let contents = format!(
        "```gleam
{formatted_type}
```
{documentation}"
    );
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(fun.location, &line_numbers)),
    }
}

fn hover_for_function_argument(argument: &Arg<Arc<Type>>, line_numbers: LineNumbers) -> Hover {
    let type_ = Printer::new().pretty_print(&argument.type_, 0);
    let contents = format!("```gleam\n{type_}\n```");
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(argument.location, &line_numbers)),
    }
}

fn hover_for_module_constant(
    constant: &ModuleConstant<Arc<Type>, EcoString>,
    line_numbers: LineNumbers,
) -> Hover {
    let empty_str = EcoString::from("");
    let type_ = Printer::new().pretty_print(&constant.type_, 0);
    let documentation = constant.documentation.as_ref().unwrap_or(&empty_str);
    let contents = format!("```gleam\n{type_}\n```\n{documentation}");
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(constant.location, &line_numbers)),
    }
}

fn hover_for_expression(expression: &TypedExpr, line_numbers: LineNumbers) -> Hover {
    let documentation = expression.get_documentation().unwrap_or_default();

    // Show the type of the hovered node to the user
    let type_ = Printer::new().pretty_print(expression.type_().as_ref(), 0);
    let contents = format!(
        "```gleam
{type_}
```
{documentation}"
    );
    Hover {
        contents: HoverContents::Scalar(MarkedString::String(contents)),
        range: Some(src_span_to_lsp_range(expression.location(), &line_numbers)),
    }
}

// Check if the inner range is included in the outer range.
// inclusion can be partial or full
fn range_includes(outer: &lsp_types::Range, inner: &lsp_types::Range) -> bool {
    (outer.start >= inner.start && outer.start <= inner.end)
        || (outer.end >= inner.start && outer.end <= inner.end)
        || (inner.start >= outer.start && inner.end <= outer.end)
}

fn code_action_unused_imports(
    module: &Module,
    params: &lsp::CodeActionParams,
    actions: &mut Vec<CodeAction>,
) {
    let uri = &params.text_document.uri;
    let unused = &module.ast.type_info.unused_imports;

    if unused.is_empty() {
        return;
    }

    // Convert src spans to lsp range
    let line_numbers = LineNumbers::new(&module.code);
    let mut hovered = false;
    let mut edits = Vec::with_capacity(unused.len());

    for unused in unused {
        let range = src_span_to_lsp_range(*unused, &line_numbers);
        // Keep track of whether any unused import has is where the cursor is
        hovered = hovered || range_includes(&params.range, &range);

        edits.push(lsp_types::TextEdit {
            range,
            new_text: "".into(),
        });
    }

    // If none of the imports are where the cursor is we do nothing
    if !hovered {
        return;
    }
    edits.sort_by_key(|edit| edit.range.start);

    CodeActionBuilder::new("Remove unused imports")
        .kind(lsp_types::CodeActionKind::QUICKFIX)
        .changes(uri.clone(), edits)
        .preferred(true)
        .push_to(actions);
}

fn gleam_pipeline_suggestions(
    module: &Module,
    params: &lsp::CodeActionParams,
    actions: &mut Vec<CodeAction>,
) {
    //dbg!(&module);
    let uri = &params.text_document.uri;
    let line_numbers = LineNumbers::new(&module.code);

    let functions = module
        .ast
        .definitions
        .iter()
        .filter(|def| def.is_function());

    let mut edits = Vec::new();

    //Check each func_body definition for possible pipeline suggestions
    for function_def in functions {
        if let Definition::Function(function) = function_def {

            let inlined_statements = inline_statements(&function.body);

            for st in &inlined_statements{
                dbg!(st);
            }

            //Create Pipeline edit
            if let Some(chains) = detect_pipeline(&inlined_statements){
                //DESTRUCTURE TUPLE
                for chain in chains {

                    let statement_for_pipeline = chain.0;

                    //Misschien bij die range check of de statements onderdeel zijn geworden van de pipeline
                    let range = src_span_to_lsp_range(statement_for_pipeline.location(), &line_numbers);
                    if range_includes(&params.range, &range){
                        let assignment_string = translate_statement_to_string(statement_for_pipeline);
                        let translated_chain = translate_func_chain_to_pipeline(chain.1);

                        //edits for the consumed statements
                        if let Some(consumed_to_be_deleted) = chain.2{
                            for src in consumed_to_be_deleted{
                                let range = src_span_to_lsp_range(src, &line_numbers);
                                edits.push(lsp_types::TextEdit {
                                        range,
                                        new_text: "".into(),
                                });
                            }
                        }

                        //edit for the pipeline convertion
                        edits.push(build_edits_from_translation(range, format!("{}{}", assignment_string, translated_chain)));
                    }
                }
            }

            //Create edit to remove the statements which are converted to inline values
            // for span_statement in statements_consumed{
            //     let range = src_span_to_lsp_range(span_statement, &line_numbers);
            //     edits.push(lsp_types::TextEdit {
            //         range,
            //         new_text: "".into(),
            //     });
            // }

        }
    }

    if !edits.is_empty(){
        CodeActionBuilder::new("Gleam Pipeline suggestion")
            .kind(lsp_types::CodeActionKind::QUICKFIX)
            .changes(uri.clone(), edits)
            .preferred(true)
            .push_to(actions)
    }

}

fn inline_statements(statements: &vec1::Vec1<Statement<Arc<Type>, TypedExpr>>) -> (Vec<Inlined>) {
    let mut inlined_statements = Vec::new();
    
    for statement in statements.iter(){
        inline_statement(statement, &mut inlined_statements)
    }

    (inlined_statements)
}

fn inline_statement(statement: &Statement<Arc<Type>, TypedExpr>, new:&mut Vec<Inlined>) {
    let mut clone = statement.clone();

    let mut expr = match &mut clone{
        Statement::Expression(e) => e,
        Statement::Assignment(a) => &mut a.value,
        _ => todo!(),
    };

    let consumed = do_the_inlining(expr, new); 

    new.push(Inlined{ statement: clone, spans_consumed_statements: consumed });
}
// fn inline_statements(statements: &vec1::Vec1<Statement<Arc<Type>, TypedExpr>>) -> (Vec<SrcSpan>, Vec<Inlined>) {
//     let mut inlined_statements = Vec::new();
//     let mut pos_consumed_statements = Vec::new();
    
//     for statement in statements.iter(){
//         inline_statement(statement, &mut inlined_statements, &mut pos_consumed_statements)
//     }

//     (pos_consumed_statements, inlined_statements)
// }

// fn inline_statement(statement: &Statement<Arc<Type>, TypedExpr>, new:&mut Vec<Inlined>, positions_to_be_emptied: &mut Vec<SrcSpan>) {
//     let mut clone = statement.clone();

//     let mut expr = match &mut clone{
//         Statement::Expression(e) => e,
//         Statement::Assignment(a) => &mut a.value,
//         _ => todo!(),
//     };

//     let consumed = do_the_inlining(expr, new); 

//     new.push(Inlined{ statement: clone, spans_consumed_statements: consumed });
// }

fn do_the_inlining(
    expr: &mut TypedExpr,
    already_inlined_statements: &mut Vec<Inlined>,
) -> Option<Vec<SrcSpan>> {
    if let TypedExpr::Call { location: _, typ: _, fun: _, args } = expr {
        let mut inlined_statements = Vec::new();

        for arg in args {
            if let TypedExpr::Call { .. } = &mut arg.value {
                if let Some(inlined) =
                    do_the_inlining(&mut arg.value, already_inlined_statements)
                {
                    inlined_statements.extend(inlined);
                }
            } else if let TypedExpr::Var {
                location: _,
                constructor,
                name: _,
            } = &mut arg.value
            {
                if let ValueConstructorVariant::LocalVariable { location } =
                    &mut constructor.variant
                {
                    let found = already_inlined_statements
                        .iter()
                        .position(|inlined| match &inlined.statement {
                            Statement::Assignment(assignment) => {
                                let assign_location = assignment.pattern.location();
                                assign_location.start == location.start
                                    && assign_location.end == location.end
                            }
                            _ => false,
                        });

                    if let Some(index) = found {
                        let statement_to_be_inlined = already_inlined_statements.remove(index);
                        
                        if let Statement::Assignment(expr_to_be_inlined) =
                        &statement_to_be_inlined.statement
                        {
                            arg.value = *expr_to_be_inlined.value.clone();
                        }
                        if let Some(already_consumed_statements) = statement_to_be_inlined.spans_consumed_statements{
                            inlined_statements.extend(already_consumed_statements);
                        }
                        inlined_statements.push(statement_to_be_inlined.statement.location());
                    }
                }
            }
        }

        if inlined_statements.is_empty() {
            None
        } else {
            Some(inlined_statements)
        }
    } else {
        None
    }
}

// fn do_the_inlining(expr: &mut TypedExpr, already_inlined_statements: &mut Vec<Statement<Arc<Type>, TypedExpr>>, position_to_be_emptied: &mut Vec<SrcSpan>) {
//     if let TypedExpr::Call { location:_, typ:_, fun:_, args } = expr{
//         for arg in args{
//             if let TypedExpr::Call { .. } = &mut arg.value{
//                 do_the_inlining(&mut arg.value, already_inlined_statements, position_to_be_emptied)
//             }

//             if let TypedExpr::Var { location: _, constructor, name: _ } = &mut arg.value{
//                 if let ValueConstructorVariant::LocalVariable { location } = &mut constructor.variant{
//                     let found = already_inlined_statements.iter().position(|statement| {
//                         if let Statement::Assignment(assignment) = statement{
//                            let assign_location = assignment.pattern.location();
//                            assign_location.start == location.start && assign_location.end == location.end
//                         } else {
//                             false
//                         }
//                     });

//                     if let Some(index) = found{
//                         let statement_to_be_inlined = already_inlined_statements.remove(index);
//                         position_to_be_emptied.push(statement_to_be_inlined.location());
//                         if let Statement::Assignment(expr_to_be_inlined) = statement_to_be_inlined{
//                             arg.value = *expr_to_be_inlined.value
//                         }
//                     }
//                 }
//             }
//         }
//     }
// }

fn translate_statement_to_string(statement: &Statement<Arc<Type>, TypedExpr>) -> String {
    if let Statement::Assignment(assign) = statement{
        match &assign.pattern{
            ast::Pattern::Int { location, value } => todo!(),
            ast::Pattern::Float { location, value } => todo!(),
            ast::Pattern::String { location, value } => todo!(),
            ast::Pattern::Var { location, name, type_ } => format!("let {} =\n", name),
            ast::Pattern::VarUsage { location, name, constructor, type_ } => todo!(),
            ast::Pattern::Assign { name, location, pattern } => todo!(),
            ast::Pattern::Discard { name, location, type_ } => todo!(),
            ast::Pattern::List { location, elements, tail, type_ } => todo!(),
            ast::Pattern::Constructor { location, name, arguments, module, constructor, with_spread, type_ } => todo!(),
            ast::Pattern::Tuple { location, elems } => todo!(),
            ast::Pattern::BitArray { location, segments } => todo!(),
            ast::Pattern::Concatenate { location, left_location, left_side_assignment, right_location, left_side_string, right_side_assignment } => todo!(),
        }
    } else{
        "".into()
    }
}

fn detect_pipeline(inlined_statements: &Vec<Inlined>) -> Option<Vec<(&Statement<Arc<Type>, TypedExpr>, Vec<&TypedExpr>, Option<Vec<SrcSpan>>)>> {

    //WAAROM EEN TUPLE???
    let mut chains_to_be_converted: Vec<(&Statement<Arc<Type>, TypedExpr>, Vec<&TypedExpr>, Option<Vec<SrcSpan>>)> = Vec::new();

    //kijken of er func_chaining plaatsvind in de argumenten
    for inlined in inlined_statements{
        let mut func_chain: Vec<&TypedExpr> = Vec::new();

        match &inlined.statement{
            Statement::Expression(expr) => retrieve_call_chain(&expr, &mut func_chain),
            Statement::Assignment(assign) => retrieve_call_chain(&assign.value, &mut func_chain),
            Statement::Use(_) => todo!(),
        }

        if !func_chain.is_empty() {
            chains_to_be_converted.push((&inlined.statement, func_chain, inlined.spans_consumed_statements.to_owned()));
        }
    }

    let result: Vec<_> = chains_to_be_converted.iter().filter(|chain| chain.1.len() > 1).collect();

    if result.is_empty(){
        None
    } else{
        Some(chains_to_be_converted)
    }
}
// fn detect_pipeline(inlined_statements: &Vec<Statement<Arc<Type>, TypedExpr>>) -> Option<Vec<(&Statement<Arc<Type>, TypedExpr>, Vec<&TypedExpr>)>> {

//     //WAAROM EEN TUPLE???
//     let mut chains_to_be_converted: Vec<(&Statement<Arc<Type>, TypedExpr>, Vec<&TypedExpr>)> = Vec::new();

//     //kijken of er func_chaining plaatsvind in de argumenten
//     for statement in inlined_statements{
//         let mut func_chain: Vec<&TypedExpr> = Vec::new();

//         match statement{
//             Statement::Expression(expr) => retrieve_call_chain(expr, &mut func_chain),
//             Statement::Assignment(assign) => retrieve_call_chain(&assign.value, &mut func_chain),
//             Statement::Use(_) => todo!(),
//         }

//         if !func_chain.is_empty() {
//             chains_to_be_converted.push((statement, func_chain));
//         }
//     }

//     let result: Vec<_> = chains_to_be_converted.iter().filter(|chain| chain.1.len() > 1).collect();

//     if result.is_empty(){
//         None
//     } else{
//         Some(chains_to_be_converted)
//     }
// }

fn translate_func_chain_to_pipeline(
    mut chains: Vec<&TypedExpr>,
) -> String {
    chains.reverse();
    let mut pipeline_format_parts: Vec<String> = Vec::new();

    if let Some(&chain) = chains.first() {
        match chain{
            TypedExpr::Call { location: _, typ: _, fun: _, args } => {

                if let Some(callarg) = args.first() {
                    //call expressie heeft WEL calargumenten, gebruik dan het eerste argument om aan te wenden als input pipeline
                    pipeline_format_parts.push(callarg.value.to_string());
                    let skinned_expr = remove_first_arg(chain);
                    pipeline_format_parts.push(skinned_expr.to_string());
                    
                } else{
                    //call expressie heeft GEEN callargumenten, dan moet de gehele call als input gebrukt worden.
                    pipeline_format_parts.push(chain.to_string());
                }
            },
            _ => todo!()
        }
    }

    for chain in chains.iter().skip(1) {
        let skinned_expr = remove_first_arg(&chain);

        dbg!(&skinned_expr);
        pipeline_format_parts.push(skinned_expr.to_string());
    }

    dbg!(format_to_pipeline(pipeline_format_parts))
}

fn format_to_pipeline(pipeline_format_parts: Vec<String>) -> String {
    let formatted_to_pipeline: String = pipeline_format_parts
    .iter()
    .enumerate()
    .map(|(index, part)| {
        if index > 0 {
            format!("|> {}", part)
        } else{
            format!("{}", part.to_string())
        }
    })
    .collect::<Vec<String>>()
    .join("\n");

    formatted_to_pipeline
}

fn remove_first_arg(parent: &TypedExpr) -> TypedExpr {
    //ombouwen naar elke keer het eerste argument eruit halen
    dbg!(parent);
    if let TypedExpr::Call {
        location,
        typ,
        fun,
        args,
    } = parent.clone()
    {
        let new_args = args.iter().skip(1).cloned().collect();

        TypedExpr::Call {
            location,
            typ,
            fun,
            args: new_args,
        }
    } else {
        // Handle other variants or unexpected types
        todo!()
    }
}

fn build_edits_from_translation(
    range: lsp_types::Range,
    translated_chain: String,
) -> lsp_types::TextEdit {
    lsp_types::TextEdit {
        range,
        new_text: translated_chain,
    }
}

fn retrieve_call_chain<'a>(expr: &'a TypedExpr, func_chain: &mut Vec<&'a TypedExpr>) {
    if let TypedExpr::Call { location:_, typ:_, fun:_, args } = expr{
        func_chain.push(&expr);

        if let Some(callarg) = args.iter().find(|callarg| {
            if let TypedExpr::Call { .. } = &callarg.value{
                true
            } else{
                false
            }
        }){
            retrieve_call_chain(&callarg.value, func_chain);
        }
    }
}

#[derive(Debug)]
struct Inlined<>{
    statement: Statement<Arc<Type>, TypedExpr>,
    spans_consumed_statements: Option<Vec<SrcSpan>>
}