use crate::{
    ast::{
        self, Arg, Definition, Function, Import, ModuleConstant, SrcSpan, TypedDefinition,
        TypedExpr, TypedPattern,
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
use std::sync::Arc;
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
fn range_includes(outer: &lsp_types::Range, inner: &lsp_types::Range) -> bool {
    (outer.start >= inner.start && outer.start <= inner.end)
        || (outer.end >= inner.start && outer.end <= inner.end)
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
    let uri = &params.text_document.uri;
    let line_numbers = LineNumbers::new(&module.code);

    let functions = module
        .ast
        .definitions
        .iter()
        .filter(|def| def.is_function());

    let mut edits = Vec::new();

    //voor elke functie definitie kijken of er gechained wordt
    for function_def in functions {
        if let Definition::Function(function) = function_def {
              //in de body van die functie kijken of er een call value als callarg is
            if let Some(chains) = detect_possible_pipeline_suggestion(&function.body){
                for chain in chains {
                    let translated_string = translate_func_chain_to_pipeline(chain);
                    edits.extend(build_edits_from_translation(translated_string));
                }
            }
        }
    }

    if !edits.is_empty(){
        CodeActionBuilder::new("Gleam Pipeline suggestion")
            .kind(lsp_types::CodeActionKind::QUICKFIX)
            .changes(uri.clone(), edits)
            .preferred(true)
            .push_to(actions)
    }

    //hier de edits bouwen?
    // let edits: Vec<lsp_types::TextEdit> = functions
    //     .filter_map(|def| {
    //         if let Definition::Function(func) = def {
    //             Some(func.body.iter().filter_map(|s| match s {
    //                 crate::ast::Statement::Expression(_) => None,
    //                 crate::ast::Statement::Assignment(assign) => {
    //                     let range = src_span_to_lsp_range(assign.location, &line_numbers);
    //                     if !range_includes(&params.range, &range) {
    //                         None
    //                     } else if let Some(edit) = suggest_pipeline_if_function_chaining(assign) {
    //                         Some(lsp_types::TextEdit {
    //                             range,
    //                             new_text: edit,
    //                         })
    //                     } else {
    //                         None
    //                     }
    //                 }
    //                 _ => None,
    //             }))
    //         } else {
    //             None
    //         }
    //     })
    //     .flatten()
    //     .collect();

    // if edits.len() > 0 {
    //     CodeActionBuilder::new("Gleam Pipeline suggestion")
    //         .kind(lsp_types::CodeActionKind::QUICKFIX)
    //         .changes(uri.clone(), edits)
    //         .preferred(true)
    //         .push_to(actions)
    // }
}

fn detect_possible_pipeline_suggestion<'a>(body: &'a vec1::Vec1<ast::Statement<Arc<Type>, TypedExpr>>) -> Option<Vec<Vec<&'a TypedExpr>>> {

    let mut chains_to_be_converted: Vec<Vec<&'a TypedExpr>> = Vec::new();

    for statement in body.iter(){
        match statement {
            ast::Statement::Expression(expression) => todo!(),
            ast::Statement::Assignment(assignment) => {
                let mut func_chain = Vec::new();
                retrieve_call_chain(&assignment.value.as_ref(), &mut func_chain);
                chains_to_be_converted.push(func_chain);
            },
            ast::Statement::Use(_) => todo!(),
        }
    }

    dbg!(&chains_to_be_converted);

    let result:Vec<&Vec<&TypedExpr>> = chains_to_be_converted.iter().filter(|chain| chain.len() > 1).collect();

    if result.is_empty(){
        None
    } else{
        Some(chains_to_be_converted)
    }
}

fn translate_func_chain_to_pipeline(
    mut chains: Vec<&TypedExpr>,
) -> String {
    chains.reverse();
    let mut pipeline_format_parts: Vec<String> = Vec::new();
    let mut location_expr_callarg_to_be_deleted: Option<&SrcSpan> = None;
    let mut previous_expr: Option<&TypedExpr> = None;

    if let Some(&chain) = chains.first() {

        match chain{
            TypedExpr::Call { location, typ, fun, args } => {

                if let Some(callarg) = args.first() {
                    //call expressie heeft WEL calargumenten, gebruik dan het eerste argument om aan te wenden als input pipeline
                    pipeline_format_parts.push(callarg.value.to_string());
                    let skinned_expr = remove_expr_from_arg_by_location(chain, &callarg.location);
                    pipeline_format_parts.push(skinned_expr.to_string());
                    previous_expr = Some(chain);
                    
                } else{
                    //call expressie heeft GEEN callargumenten, dan moet de gehele call als input gebrukt worden.
                    pipeline_format_parts.push(chain.to_string());
                }
            },
            _ => todo!()
        }
       
    }

    for chain in chains.iter().skip(1) {
        //remove expr from callarg 
        let skinned_expr = remove_expr_from_arg_by_location(&chain, &previous_expr.unwrap().location());
        pipeline_format_parts.push(skinned_expr.to_string());
        previous_expr = Some(chain);
    }

    dbg!(format_to_pipeline(pipeline_format_parts))
}

fn format_to_pipeline(pipeline_format_parts: Vec<String>) -> String {
    // let formatted_to_pipeline = pipeline_format_parts.join("|>");
    let formatted_to_pipeline: String = pipeline_format_parts
    .iter()
    .enumerate()
    .map(|(index, part)| {
        if index > 0 {
            format!("\n|>{}", part)
        } else{
            part.to_string()
        }
    })
    .collect::<Vec<String>>()
    .join("\n");

    dbg!(formatted_to_pipeline)
}

fn remove_expr_from_arg_by_location(parent: &TypedExpr, location_arg:&SrcSpan ) -> TypedExpr {
    if let TypedExpr::Call {
        location,
        typ,
        fun,
        mut args,
    } = parent.clone()
    {
        let new_args = args
            .iter()
            .filter_map(|arg| {
                if arg.location.start != location_arg.start || arg.location.end != location_arg.end {
                    Some(arg.clone()) // Keep the arguments whose location doesn't match
                } else {
                    None // Discard the argument whose location matches
                }
            })
            .collect();

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

// fn translate_callarg_to_string(callarg: &CallArg<TypedExpr>) -> String {
//     &callarg.value
// }
// fn translate_expr_to_string(
//     expr: &TypedExpr
// ) -> String {
    // match expr{
    //     TypedExpr::Int { location, typ, value } => todo!(),
    //     TypedExpr::Float { location, typ, value } => todo!(),
    //     TypedExpr::String { location, typ, value } => todo!(),
    //     TypedExpr::Block { location, statements } => todo!(),
    //     TypedExpr::Pipeline { location, assignments, finally } => todo!(),
    //     TypedExpr::Var { location, constructor, name } => todo!(),
    //     TypedExpr::Fn { location, typ, is_capture, args, body, return_annotation } => todo!(),
    //     TypedExpr::List { location, typ, elements, tail } => {
    //         let mut translation: String = "[".into();

    //         for (index, element) in elements.iter().enumerate() {
    //             if index > 0 {
    //                 translation.push_str(", ");
    //             }
    //             translation.push_str(&element.to_string());
    //         }
        
    //         translation.push(']');
        
    //         translation

    //     },
    //     TypedExpr::Call { location, typ, fun, args } => "fn () {}".into(),
    //     TypedExpr::BinOp { location, typ, name, left, right } => todo!(),
    //     TypedExpr::Case { location, typ, subjects, clauses } => todo!(),
    //     TypedExpr::RecordAccess { location, typ, label, index, record } => todo!(),
    //     TypedExpr::ModuleSelect { location, typ, label, module_name, module_alias, constructor } => todo!(),
    //     TypedExpr::Tuple { location, typ, elems } => todo!(),
    //     TypedExpr::TupleIndex { location, typ, index, tuple } => todo!(),
    //     TypedExpr::Todo { location, message, type_ } => todo!(),
    //     TypedExpr::Panic { location, message, type_ } => todo!(),
    //     TypedExpr::BitArray { location, typ, segments } => todo!(),
    //     TypedExpr::RecordUpdate { location, typ, spread, args } => todo!(),
    //     TypedExpr::NegateBool { location, value } => todo!(),
    //     TypedExpr::NegateInt { location, value } => todo!(),
    // }
// }

fn build_edits_from_translation(
    translated_chains: String,
) -> Vec<lsp_types::TextEdit> {
    // Implementation goes here
    todo!()
}

fn retrieve_call_chain<'a>(expr: &'a TypedExpr, func_chain: &mut Vec<&'a TypedExpr>) {
    if let TypedExpr::Call { location, typ, fun, args } = expr{
        func_chain.push(&expr);

        if let Some(callarg) = args.iter().find(|callarg| {
            if let TypedExpr::Call { location, typ, fun, args } = &callarg.value{
                true
            } else{
                false
            }
        }){
            dbg!(&callarg);
            retrieve_call_chain(&callarg.value, func_chain);
        }
    }
}

fn suggest_pipeline_if_function_chaining(
    assign: &crate::ast::Assignment<Arc<Type>, TypedExpr>,
) -> Option<String> {
    if let TypedExpr::Call {
        location,
        typ,
        fun,
        args,
    } = assign.value.as_ref()
    {
        Some("|>".into())
    } else {
        None
    }
}

fn detect_func_chaining_in_assign(
    assign: &crate::ast::Assignment<Arc<Type>, TypedExpr>,
) -> Option<&crate::ast::Assignment<Arc<Type>, TypedExpr>> {
    
    if let TypedExpr::Call {
        location: _,
        typ: _,
        fun: _,
        args,
    } = assign.value.as_ref()
    {
        if args.iter().any(|arg| matches!(arg.value, TypedExpr::Call { .. })) {
            Some(assign)
        } else{
            None
        }
    } else {
        None
    }
}

// fn func_chaining_in_func_arg(call: &TypedExpr, mut call_chain: Vec<TypedExpr>) -> Vec<TypedExpr> {
//     if let TypedExpr::Call {
//         location,
//         typ,
//         fun,
//         args,
//     } = call
//     {
//         if args.iter().any(|arg| {
//             if let TypedExpr::Call {
//                 location,
//                 typ,
//                 fun,
//                 args,
//             } = &arg.value
//             {
//                 true
//             } else {
//                 false
//             }
//         }) {
//             // Add the current call to the call_chain vector
//             call_chain.push(call.clone());

//             // Recursively continue with each argument
//             for arg in args {
//                 call_chain = func_chaining_in_func_arg(&arg.value, call_chain);
//             }
//         }
//     }

//     call_chain
// }
