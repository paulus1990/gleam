use crate::{
    build::{ErlangAppCodegenConfiguration, Module},
    config::PackageConfig,
    erlang,
    io::FileSystemWriter,
    javascript,
    line_numbers::LineNumbers,
    Result, ast::{BinOp}, warning::{WarningEmitter, TypeWarningEmitter},
};
use ecow::EcoString;
use im::HashMap;
use itertools::Itertools;
use wast::{token::{NameAnnotation, Span, Id, Index}, core::{Func, FuncKind, ValType, Expression, Instruction, InlineExport, TypeUse, FunctionType}, Error};
use std::{fmt::Debug, sync::Arc, borrow::Borrow, mem, fs::File, io::Write};
use std::cell::RefCell;

use camino::Utf8Path;
use wast::core::{ModuleField, ModuleKind};
use wast::core::Instruction::I64Add;
use crate::analyse::TargetSupport;
use crate::ast::{ArgNames, Definition, Function, Statement, TypedExpr};
use crate::type_::Type;
// use crate::ast::{Definition, TypedExpr};
// use crate::type_::{ModuleInterface, Type};
//
//
// // #[derive(Debug)]
// // pub struct Wasm<'a> {
// //     // build_directory: &'a Utf8Path,
// //     // include_directory: &'a Utf8Path,
// // }
//
//
// // impl<'a> Wasm<'a> {
// //     pub fn render<Writer: FileSystemWriter>(
// //         &self,
// //         writer: Writer,
// //         modules: &[Module],
// //     ) -> Result<()> {}
// // }
//
//
// //TODO sure String is terrible etc. etc. can imporve later. And it's actually WAT lol.
// fn wasm_definition(statement: crate::ast::TypedDefinition) -> String {
//     let mut ret = String::new();
//
//     // let def: TypedDefinition = statement.definition;
//     match statement {
//         crate::ast::Definition::Function(crate::ast::Function { location, end_position, name, arguments, body, public, deprecation, return_annotation, return_type, documentation, external_erlang, external_javascript, supported_targets }) => {
//                 let mut params = String::new();
//                 // dbg!(&arguments);
//                 // let len = arguments.len();
//                 // dbg!(len);
//                 for param in arguments {
//                     dbg!(&param.names);
//
//                     let name = match &param.names {
//                         crate::ast::ArgNames::Discard { name } => todo!(),
//                         crate::ast::ArgNames::LabelledDiscard { label, name } => todo!(),
//                         crate::ast::ArgNames::Named { name } => name,
//                         crate::ast::ArgNames::NamedLabelled { name, label } => todo!(),
//                     };
//
//                     //TODO duplicated
//                     let type_ = match param.type_.as_ref() {
//                         crate::type_::Type::Named { public, module, name, args } if name == "Int" => {
//                             "i32"
//                         },
//                         crate::type_::Type::Named { public, module, name, args } => todo!(),
//                         crate::type_::Type::Fn { args, retrn } => todo!(),
//                         crate::type_::Type::Var { type_ } => todo!(),
//                         crate::type_::Type::Tuple { elems } => todo!(),
//                     };
//
//                     params = format!("{params} (param ${name} {type_})");
//                 }
//
//
//                 // let result = match return_annotation {
//                 //     Some(crate::ast::TypeAst::Var(crate::ast::TypeAstVar{
//                 //         name,
//                 //         location
//                 //     })) => {
//                 //         // TODO hmmm the name is a string is that the type info?
//                 //         "(result {name})"
//                 //     },
//                 //     None => "",
//                 //     _ => todo!()
//                 // };
//                 let hmm: std::sync::Arc<crate::type_::Type> = return_type.clone(); // Why is it unit here but &Arc<Type> in typescript? Cause that was a typed module, now here too
//                 dbg!(hmm);
//
//
//
//                 let mut result = "";
//
//
//                 match return_type.as_ref() {
//                     crate::type_::Type::Named {
//                         module,name,public,args
//                         // module: ecow::EcoString::from("Gleam"),
//                         // name: "Int",
//                         // ..
//                     } => {
//                         if name.eq_ignore_ascii_case("int") && module.eq_ignore_ascii_case("gleam") {
//                             // TODO the type matching for sure needs abstraction I think the other backends do preprocessing and maybe create a mapping or something?
//                             // at least also create (make the correct representation I think) the custom types etc.
//                             // Also need to check is a Gleam::Int really a i32
//                             // Might be a step check the Gleam module types and map them to WASM, good idea to build on for custom types
//                             result = "(result i32)"
//                         } else {
//                             todo!()
//                         }
//                     },
//                     _ => todo!()
//                 }
//
//                 let mut func_body = String::new();
//
//
//                 dbg!(&body);
//
//                 for expr in body {
//                    let expr_s =  match expr {
//                         crate::ast::Statement::Expression(expr) => {
//                             match expr {
//                                 crate::ast::TypedExpr::BinOp { location, typ, name, left, right } if name == BinOp::AddInt => {
//                                     let lhs = match left.as_ref() {
//                                         crate::ast::TypedExpr::Var { location, constructor, name } => {format!("(local.get ${name})")},
//                                         crate::ast::TypedExpr::Int { location, typ, value } => todo!(),
//                                         crate::ast::TypedExpr::Float { location, typ, value } => todo!(),
//                                         crate::ast::TypedExpr::String { location, typ, value } => todo!(),
//                                         crate::ast::TypedExpr::Block { location, statements } => todo!(),
//                                         crate::ast::TypedExpr::Pipeline { location, assignments, finally } => todo!(),
//                                         crate::ast::TypedExpr::Fn { location, typ, is_capture, args, body, return_annotation } => todo!(),
//                                         crate::ast::TypedExpr::List { location, typ, elements, tail } => todo!(),
//                                         crate::ast::TypedExpr::Call { location, typ, fun, args } => todo!(),
//                                         crate::ast::TypedExpr::BinOp { location, typ, name, left, right } => todo!(),
//                                         crate::ast::TypedExpr::Case { location, typ, subjects, clauses } => todo!(),
//                                         crate::ast::TypedExpr::RecordAccess { location, typ, label, index, record } => todo!(),
//                                         crate::ast::TypedExpr::ModuleSelect { location, typ, label, module_name, module_alias, constructor } => todo!(),
//                                         crate::ast::TypedExpr::Tuple { location, typ, elems } => todo!(),
//                                         crate::ast::TypedExpr::TupleIndex { location, typ, index, tuple } => todo!(),
//                                         crate::ast::TypedExpr::Todo { location, message, type_ } => todo!(),
//                                         crate::ast::TypedExpr::Panic { location, message, type_ } => todo!(),
//                                         crate::ast::TypedExpr::BitArray { location, typ, segments } => todo!(),
//                                         crate::ast::TypedExpr::RecordUpdate { location, typ, spread, args } => todo!(),
//                                         crate::ast::TypedExpr::NegateBool { location, value } => todo!(),
//                                         crate::ast::TypedExpr::NegateInt { location, value } => todo!(),
//                                     };
//
//
//                                     // TODO obv duplication from above
//                                     let rhs = match right.as_ref() {
//                                         crate::ast::TypedExpr::Var { location, constructor, name } => {format!("(local.get ${name})")},
//                                         _ => todo!()
//                                     };
//
//                                     format!("(i32.add {lhs} {rhs})")
//                                 },
//                                 crate::ast::TypedExpr::Int { location, typ, value } => todo!(),
//                                 crate::ast::TypedExpr::Float { location, typ, value } => todo!(),
//                                 crate::ast::TypedExpr::String { location, typ, value } => todo!(),
//                                 crate::ast::TypedExpr::Block { location, statements } => todo!(),
//                                 crate::ast::TypedExpr::Pipeline { location, assignments, finally } => todo!(),
//                                 crate::ast::TypedExpr::Var { location, constructor, name } => todo!(),
//                                 crate::ast::TypedExpr::Fn { location, typ, is_capture, args, body, return_annotation } => todo!(),
//                                 crate::ast::TypedExpr::List { location, typ, elements, tail } => todo!(),
//                                 crate::ast::TypedExpr::Call { location, typ, fun, args } => todo!(),
//                                 crate::ast::TypedExpr::BinOp { location, typ, name, left, right } => todo!(),
//                                 crate::ast::TypedExpr::Case { location, typ, subjects, clauses } => todo!(),
//                                 crate::ast::TypedExpr::RecordAccess { location, typ, label, index, record } => todo!(),
//                                 crate::ast::TypedExpr::ModuleSelect { location, typ, label, module_name, module_alias, constructor } => todo!(),
//                                 crate::ast::TypedExpr::Tuple { location, typ, elems } => todo!(),
//                                 crate::ast::TypedExpr::TupleIndex { location, typ, index, tuple } => todo!(),
//                                 crate::ast::TypedExpr::Todo { location, message, type_ } => todo!(),
//                                 crate::ast::TypedExpr::Panic { location, message, type_ } => todo!(),
//                                 crate::ast::TypedExpr::BitArray { location, typ, segments } => todo!(),
//                                 crate::ast::TypedExpr::RecordUpdate { location, typ, spread, args } => todo!(),
//                                 crate::ast::TypedExpr::NegateBool { location, value } => todo!(),
//                                 crate::ast::TypedExpr::NegateInt { location, value } => todo!(),
//                             }
//                         },
//                         crate::ast::Statement::Assignment(_) => todo!(),
//                         crate::ast::Statement::Use(_) => todo!(),
//                     };
//                     func_body = format!("{func_body}\n{expr_s}");
//                 }
//
//                 ret = std::format!("(func ${name} {params} {result}
//                     {func_body}
//                 )");
//
//
//         },
//         crate::ast::Definition::TypeAlias(_) => todo!(),
//         crate::ast::Definition::CustomType(_) => todo!(),
//         crate::ast::Definition::Import(_) => todo!(),
//         crate::ast::Definition::ModuleConstant(_) => todo!(),
//     }
//
//     ret
//
// }
//
// #[test]
// fn wasm_1st() {
//     // cargo test --package gleam-core --lib -- codegen::wasm_1st --exact --nocapture
//     let module = trying_to_make_module();
//
//
//     // running 1 test
//     // thread 'codegen::wasm_1st' panicked at 'Unable to find prelude in importable modules', compiler-core/src/type_/environment.rs:86:14
//     // note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace
//     // test codegen::wasm_1st ... FAILED
// // lol, need some prelude....................... to import. Interesting.. Has to have name Gleam and actually seems to supply types like Gleam::Int -> TODO check out
//
//
//     let defs = module.definitions;
//     let len = &defs.len();
//     println!("{defs:?}\nlen: {len}");
//
// // Man TODO exports etc, so after the result so the function can be used externally etc. Find out if WASM has a main function concept????
// for def in defs {
//     let res = wasm_definition(def);
//     let res = format!("(module {res})");
//     println!("{res}");
//     std::fs::write("/home/harm/git/gleam/exper/hmm.wat", res).expect("filewrite");
//
//     // Dang return is strange to find... Return annotation is None namely, has to be a typed module ;)
//
//     //wat2wasm exper/hmm.wat -o exper/hmm.wasm
//
//     // So interesting the WAT is fine when turned into WASM but things happen there too. But WAT output might be useful as an option anyway
//     // (can add the source code comments to it for ex.), but making the WASM directly
//     // is a choice! We can also output the WASM directly, or re-use wat2wasm or something that optimizes too so WAT is our real ouput..... Is a choice
//
//
//     // Big things:
//     // - Imports/Modules in general, I don't understand them.
//     // - Took a hacky approach to getting the AST set-up, probably introduced blind spots this way; no holistic understanding.
//     // - Output formats, and their optimality. So really need a better understanding of WASM, will make mapping easier too.
//     // - General theory, but can move to practical sooner and let that guide? At this stage, then also check supervisor for hints here?
//
//     // Learned
//     // - A(!) way to get a fast feedback loop, but probably needs improvement since now takes shortcuts.
//     // - A wider view of the scope, the explicit todo!s in the match arms are all work todo, I mean the overarching abstractions seem not hard to abstract (allowing dedup etc.).
//     //   Like the (module ...) needs the exports etc. can easily be a different level than introducing custom types a different level again from function bodies
//     //   which is a different level again from correct type mapping to WASM etc. etc.. Fortunatly can check the ts/js/erlang existing stuff.
//     // - This way of doing it is not extensible, a change in the gelam code to subtraction is conceptually EZPZ but this way in practice hard!
//
// }
//
// }

fn trying_to_make_module() -> crate::ast::Module<crate::type_::ModuleInterface, crate::ast::Definition<Arc<crate::type_::Type>, crate::ast::TypedExpr, EcoString, EcoString>> {
    let parsed = crate::parse::parse_module(
        "pub fn add(x: Int, y: Int) -> Int {
            x + y
          }",
    )
    .expect("syntax error");
    let module = parsed.module;
    // println!("{module:?}");
    let ids = crate::uid::UniqueIdGenerator::new();
    let small = ecow::EcoString::from("Welcome");
    let mut hs = im::hashmap::HashMap::new();
    let hs2: std::collections::HashMap<EcoString, EcoString> = std::collections::HashMap::new();
    let we = TypeWarningEmitter::new(
        camino::Utf8PathBuf::new(),
        "".into(),
        WarningEmitter::new(
            Arc::new(crate::warning::VectorWarningEmitterIO::default()),
        ),
    );


    let _ = hs.insert(
        crate::type_::PRELUDE_MODULE_NAME.into(),
        crate::type_::build_prelude(&ids),
    );

    let module = crate::analyse::infer_module(crate::build::Target::JavaScript,&ids,
    module,
    crate::build::Origin::Src,&small,&hs,&we,&hs2,TargetSupport::NotEnforced).expect("type error?");
    module
}


struct WasmThing<'a> {
    gleam_module: crate::ast::Module<crate::type_::ModuleInterface, crate::ast::Definition<Arc<crate::type_::Type>, crate::ast::TypedExpr, EcoString, EcoString>>,
    wasm_instructions: RefCell<Vec<ModuleField<'a>>>, //AST
    //Id is pretty private :( identifiers: HashMap<&'a str, Id<'a>>, // Symbol table, but not really, wanted to use for wasm names but unnecessary byte code. Will matter if we do in Gleam  "let x=1; ds(x);"
    identifiers: HashMap<&'a str, usize>,
    known_types: HashMap<&'a str,ValType<'a>>
    }

fn known_types() -> HashMap<&'static str,ValType<'static>> {
    let mut map = HashMap::new();
    map.insert("Int", ValType::I64);
    map
}

impl<'a> WasmThing<'a> {
    // fn new(gleam_module: crate::ast::Module<crate::type_::ModuleInterface, crate::ast::Definition<Arc<crate::type_::Type>, crate::ast::TypedExpr, EcoString, EcoString>>) -> WasmThing<'a> {
    //     WasmThing {
    //         gleam_module,
    //         wasm_instructions: vec![],
    //         identifiers: Default::default(),
    //         known_types: known_types()
    //         // TODO prolly need types imported and a whole thing when getting some more
    //     }
    // }

    fn transform(mut self) -> std::result::Result<Vec<u8>, Error> {
        let offset = 0; //For now we pretend each is a
        // let name = self.gleam_module.name;

        for gleam_definition in &self.gleam_module.definitions {
            self.transform_gleam_definition(gleam_definition);
        }

        let mut wasm_module = wast::core::Module {
            span: Span::from_offset(offset),
            id: None,
            name: None,//Some(NameAnnotation{name: &name}), //Could be none if encoded?
            kind: ModuleKind::Text(self.wasm_instructions.take()),
        };

        wasm_module.encode()
    }

    fn transform_gleam_definition(&self, gleam_expression: &'a Definition<Arc<Type>, TypedExpr, EcoString, EcoString>) {
        match gleam_expression {
            Definition::Function(gleam_function) => {
                self.add_gleam_function_to_wasm_instructions(gleam_function);
            }
            _ => todo!()
        }
    }

    fn add_gleam_function_to_wasm_instructions(&self, gleam_function: &'a Function<Arc<Type>, TypedExpr>) {
        let offset = gleam_function.location.start as usize;
        let span = Span::from_offset(offset);
        let result_type = self.transform_gleam_type(gleam_function.return_type.as_ref());
        let mut params: Box<[(Option<Id<'a>>, Option<NameAnnotation<'a>>, ValType<'a>)]> = Box::new([]);
        let mut arguments = Vec::from(mem::take(&mut params));
        // local.get 0 always the 1st arg here? What if constants. TODO prolly new Vec here!
        let mut scope = self.identifiers.clone();
        for param in gleam_function.arguments.iter() {
            let name = self.get_gleam_name(&param.names);
            scope.insert(name, scope.len());
            let type_ = self.transform_gleam_type(param.type_.as_ref());
            arguments.push((None, None, type_));
        }
        let mut instrs: Box<[Instruction<'a>]> = Box::new([]);
        let mut instructions = Vec::from(mem::take(&mut instrs));
        for gleam_statement in gleam_function.body.iter() {
            instructions.append(&mut self.transform_gleam_statement(gleam_statement, &mut scope))
        }

        let ty = TypeUse {
            index: None,
            inline: Some(FunctionType {
                params: arguments.into(),
                results: Box::new([result_type]),
            }),
        };

        let wasm_func = Func {
            span,
            id: None,
            name: Some(NameAnnotation { name: &gleam_function.name }),
            exports: InlineExport::default(),
            kind: FuncKind::Inline {
                locals: Box::new([]), //TODO maybe get from scope, it'd bet the slice of it that's bigger than the arguments length...
                expression: Expression { instrs: instructions.into() }
            },
            ty,
        };
        self.wasm_instructions.borrow_mut().push(ModuleField::Func(wasm_func));
    }

    fn get_gleam_name(&self, names: &'a ArgNames) -> &'a str {
        match names {
            ArgNames::Named { name } => {
                return name.as_str();
            }
            _ => todo!()
        };
        todo!()
    }
    
    fn transform_gleam_statement(&self, gleam_statement: &'a Statement<Arc<Type>, TypedExpr>, scope: &mut HashMap<&str, usize>) -> Vec<Instruction<'a>> {
        match gleam_statement {
            Statement::Expression(gleam_expression) => {
                self.transform_gleam_expression(gleam_expression, scope)
            }
            _ => todo!()
        }
    }

    fn transform_gleam_expression(&self, gleam_expression: &TypedExpr, scope: &mut HashMap<&str, usize>) -> Vec<Instruction<'a>> {
        let mut instructions = Vec::new();
        match gleam_expression {
            TypedExpr::BinOp { name, left, right, .. } => {
                instructions.push(self.transform_gleam_bin_op(name));
                instructions.append(&mut self.transform_gleam_expression(left.as_ref(), scope));
                instructions.append(&mut self.transform_gleam_expression(right.as_ref(), scope));
            }
            TypedExpr::Var{name, location, ..} => {
               let idx = scope.get(name.as_str()).expect("I expect all vars to be in the scope right now."); //TODO globals different... Need some logic here to decide the local/global get if necessary
               return vec![Instruction::LocalGet(Index::Num(*idx as u32, Span::from_offset(location.start as usize)))];
            }
            _ => todo!()
        }
        instructions
    }

    fn transform_gleam_bin_op(&self, name: &BinOp) -> Instruction<'a> {
        match name {
            BinOp::AddInt => I64Add,
            _ => todo!()
        }
    }

    fn transform_gleam_type(&self, type_: &Type) -> ValType<'a> {
        match type_ {
            Type::Named { name, .. } => {
                self.known_types.get(name.as_str()).expect("For now we expect to know all types").clone()
            },
            _ => todo!()
        }
    }
}

#[test]
fn wasm_2n() {
    // use wast::core::{Module, ModuleKind,ModuleField};

    let gleam_module = trying_to_make_module();

    // let w = WasmThing::new(gleam_module);
    let w = WasmThing {
        gleam_module,
        wasm_instructions: RefCell::new(vec![]),
        identifiers: Default::default(),
        known_types: known_types()
        // TODO prolly need types imported and a whole thing when getting some more
    };
    let res = w.transform().unwrap();
    let mut file = File::create("letstry.wasm").unwrap();
    file.write_all(&res);

//     let mut module_fields = Vec::new();
//
//     for exp in gleam_module.definitions {
//         match exp{
//             crate::ast::Definition::Function(f) => {
//
//                 // let mut locals = [Local{ id: todo!(), name: todo!(), ty: todo!() };f.arguments.len()];
//                 // let mut arguments = Vec::new();
//                 let mut params: Box<[Local<'_>]> = Box::new([]);
//                 let mut arguments = Vec::from(mem::take(&mut params));
//                 for param in f.arguments.into_iter() {
//                     let ty =  match param.type_.as_ref() {
//                         crate::type_::Type::Named { public, module, name, args } => match name.as_str() {
//                             "Int" => ValType::I64, //TODO take from symbol table? So we get custom types too?
//                             _ => todo!()
//                         },
//                         crate::type_::Type::Fn { args, retrn } => todo!(),
//                         crate::type_::Type::Var { type_ } => todo!(),
//                         crate::type_::Type::Tuple { elems } => todo!(),
//                     };
//                     // let name = param.get_variable_name().map_or(None, |a| {
//                     //     let x = a.to_owned();
//                     //     Some(NameAnnotation{ name: &a})});
//                     // let mut name = None;
//
//                     // if param.get_variable_name().is_some() {
//                     //     let x = param.get_variable_name().unwrap().to_owned();
//                     //     // let anno: NameAnnotation<'a> = NameAnnotation {name: &'staticx};
//                     //     let bugger: &'static str =  &x;
//                     //     let anno = NameAnnotation {name: *bugger};
//                     //     name = Some(anno);
//                     // }
//
//                     arguments.push(Local {
//                         id: None,
//                         name: None, //Bah I want the name but the lifetime... maybe we put the lifetime so we can own here and this func has the lifetime of the whole thind, whatever for now none
//                         ty,
//                     });
//                 };
//
//                 // let mut instructions = [Expression;f.body.len()];
//                 // let mut instructions = Vec::new();
//                 // let mut params = [];
//                 // let mut arguments = Vec::from(mem::take(&mut params));
//                 let mut instrs: Box<[Instruction<'_>]> = Box::new([]);
//                 let mut instructions = Vec::from(mem::take(&mut instrs));
//                 for g_e in f.body.into_iter() {
//                     match g_e {
//                         crate::ast::Statement::Expression(inner_e) => {
//                             match inner_e {
//                                 // Today early afternoon 14/1 left-off here!
//                                 crate::ast::TypedExpr::Int { location, typ, value } => todo!(),
//                                 crate::ast::TypedExpr::Float { location, typ, value } => todo!(),
//                                 crate::ast::TypedExpr::String { location, typ, value } => todo!(),
//                                 crate::ast::TypedExpr::Block { location, statements } => todo!(),
//                                 crate::ast::TypedExpr::Pipeline { location, assignments, finally } => todo!(),
//                                 crate::ast::TypedExpr::Var { location, constructor, name } => todo!(),
//                                 crate::ast::TypedExpr::Fn { location, typ, is_capture, args, body, return_annotation } => todo!(),
//                                 crate::ast::TypedExpr::List { location, typ, elements, tail } => todo!(),
//                                 crate::ast::TypedExpr::Call { location, typ, fun, args } => todo!(),
//                                 crate::ast::TypedExpr::BinOp { location, typ, name, left, right } => {
//                                     match name {
//                                         BinOp::And => todo!(),
//                                         BinOp::Or => todo!(),
//                                         BinOp::Eq => todo!(),
//                                         BinOp::NotEq => todo!(),
//                                         BinOp::LtInt => todo!(),
//                                         BinOp::LtEqInt => todo!(),
//                                         BinOp::LtFloat => todo!(),
//                                         BinOp::LtEqFloat => todo!(),
//                                         BinOp::GtEqInt => todo!(),
//                                         BinOp::GtInt => todo!(),
//                                         BinOp::GtEqFloat => todo!(),
//                                         BinOp::GtFloat => todo!(),
//                                         BinOp::AddInt => instructions.push(Instruction::I64Add),
//                                         BinOp::AddFloat => todo!(),
//                                         BinOp::SubInt => todo!(),
//                                         BinOp::SubFloat => todo!(),
//                                         BinOp::MultInt => todo!(),
//                                         BinOp::MultFloat => todo!(),
//                                         BinOp::DivInt => todo!(),
//                                         BinOp::DivFloat => todo!(),
//                                         BinOp::RemainderInt => todo!(),
//                                         BinOp::Concatenate => todo!(),
//                                     };
//
//                                     instructions.push(Instruction::LocalGet(Index::Num(0, Span::from_offset(0)))); //TODO id is better but... ya know put it in a symbol table
//                                     instructions.push(Instruction::LocalGet(Index::Num(1, Span::from_offset(0))));
//                                     // match left {
//                                     //  //TODO!
//                                     // },
//                                 }
//                                 crate::ast::TypedExpr::Case { location, typ, subjects, clauses } => todo!(),
//                                 crate::ast::TypedExpr::RecordAccess { location, typ, label, index, record } => todo!(),
//                                 crate::ast::TypedExpr::ModuleSelect { location, typ, label, module_name, module_alias, constructor } => todo!(),
//                                 crate::ast::TypedExpr::Tuple { location, typ, elems } => todo!(),
//                                 crate::ast::TypedExpr::TupleIndex { location, typ, index, tuple } => todo!(),
//                                 crate::ast::TypedExpr::Todo { location, message, type_ } => todo!(),
//                                 crate::ast::TypedExpr::Panic { location, message, type_ } => todo!(),
//                                 crate::ast::TypedExpr::BitArray { location, typ, segments } => todo!(),
//                                 crate::ast::TypedExpr::RecordUpdate { location, typ, spread, args } => todo!(),
//                                 crate::ast::TypedExpr::NegateBool { location, value } => todo!(),
//                                 crate::ast::TypedExpr::NegateInt { location, value } => todo!(),
//                             }
//                         },
//                         crate::ast::Statement::Assignment(_) => todo!(),
//                         crate::ast::Statement::Use(_) => todo!(),
//                     };
//                 }
//
//                 let offset = f.location.start as usize;
//                 let span = Span::from_offset(offset);
//
//
//                 let results_type = match f.return_type.as_ref() {
//                     crate::type_::Type::Named { public, module, name, args } => match name.as_str() {
//                         "Int" => ValType::I64, //TODO take from symbol table? So we get custom types too?
//                         _ => todo!()
//                     },
//                     crate::type_::Type::Fn { args, retrn } => todo!(),
//                     crate::type_::Type::Var { type_ } => todo!(),
//                     crate::type_::Type::Tuple { elems } => todo!(),
//                 };
//                 //TODO here copying
//                 let mut huh: Box<[(Option<Id<'_>>, Option<NameAnnotation<'_>>, ValType<'_>)]> = Box::new([]);
//                 let mut args_procced = Vec::from(mem::take(&mut huh));;
//                 // let mut params = [];
//                 // let mut arguments = Vec::from(mem::take(&mut params));
//                 for arg in &arguments {
//                     args_procced.push((arg.id, arg.name, arg.ty))
//                 }
//
//                 let ty = TypeUse {
//                     index: None,
//                     inline: Some(FunctionType {
//                         params: args_procced.into(),
//                         results: Box::new([results_type]),
//                     }),
//                 };
//
//                 // dbg!(instructions.len());
// //Hey hmm type field automatically filled? unclear to me how tho...
//                 let wf = Func {
//                     span,
//                     id: None,
//                     name: Some(NameAnnotation{name: "dangitawrongliteral"}), //f.name Bah I want the name but the lifetime... maybe we put the lifetime so we can own here and this func has the lifetime of the whole thind, whatever for now none
//                     exports: InlineExport::default(),
//                     kind: FuncKind::Inline{
//                         locals: Box::new([]), //TODO
//                         expression: Expression { instrs:instructions.into() } },
//                     ty,
//                 };
//
//                 module_fields.push(ModuleField::Func(wf))
//             },
//             crate::ast::Definition::TypeAlias(_) => todo!(),
//             crate::ast::Definition::CustomType(_) => todo!(),
//             crate::ast::Definition::Import(_) => todo!(),
//             crate::ast::Definition::ModuleConstant(_) => todo!(),
//         }
//     }
//
//     let m_name = gleam_module.name;
//     let offset = 0;
//     let mut wasm_module = Module {
//         span: Span::from_offset(offset),
//         id: None,
//         name: Some(NameAnnotation{name: &m_name}), //TODO use a parser? But find out which they use
//         kind: ModuleKind::Text(module_fields),
//     };
//
//     //TODO hmm, looks like a module has field or blobs, inside the module kind
//     // dbg!(wasm_module.encode().unwrap());
//     // assert!(false)
//     let mut file = File::create("letstry.wasm").unwrap();
//     file.write_all(&wasm_module.encode().unwrap());
//     // Totally the wrong instructions!

}

/// A code generator that creates a .erl Erlang module and record header files
/// for each Gleam module in the package.
#[derive(Debug)]
pub struct Erlang<'a> {
    build_directory: &'a Utf8Path,
    include_directory: &'a Utf8Path,
}

impl<'a> Erlang<'a> {
    pub fn new(build_directory: &'a Utf8Path, include_directory: &'a Utf8Path) -> Self {
        Self {
            build_directory,
            include_directory,
        }
    }

    pub fn render<Writer: FileSystemWriter>(
        &self,
        writer: Writer,
        modules: &[Module],
    ) -> Result<()> {
        for module in modules {
            let erl_name = module.name.replace("/", "@");
            self.erlang_module(&writer, module, &erl_name)?;
            self.erlang_record_headers(&writer, module, &erl_name)?;
        }
        Ok(())
    }

    fn erlang_module<Writer: FileSystemWriter>(
        &self,
        writer: &Writer,
        module: &Module,
        erl_name: &str,
    ) -> Result<()> {
        let name = format!("{erl_name}.erl");
        let path = self.build_directory.join(&name);
        let line_numbers = LineNumbers::new(&module.code);
        let output = erlang::module(&module.ast, &line_numbers);
        tracing::debug!(name = ?name, "Generated Erlang module");
        writer.write(&path, &output?)
    }

    fn erlang_record_headers<Writer: FileSystemWriter>(
        &self,
        writer: &Writer,
        module: &Module,
        erl_name: &str,
    ) -> Result<()> {
        for (name, text) in erlang::records(&module.ast) {
            let name = format!("{erl_name}_{name}.hrl");
            tracing::debug!(name = ?name, "Generated Erlang header");
            writer.write(&self.include_directory.join(name), &text)?;
        }
        Ok(())
    }
}

/// A code generator that creates a .app Erlang application file for the package
#[derive(Debug)]
pub struct ErlangApp<'a> {
    output_directory: &'a Utf8Path,
    config: &'a ErlangAppCodegenConfiguration,
}

impl<'a> ErlangApp<'a> {
    pub fn new(output_directory: &'a Utf8Path, config: &'a ErlangAppCodegenConfiguration) -> Self {
        Self {
            output_directory,
            config,
        }
    }

    pub fn render<Writer: FileSystemWriter>(
        &self,
        writer: Writer,
        config: &PackageConfig,
        modules: &[Module],
    ) -> Result<()> {
        fn tuple(key: &str, value: &str) -> String {
            format!("    {{{key}, {value}}},\n")
        }

        let path = self.output_directory.join(format!("{}.app", &config.name));

        let start_module = config
            .erlang
            .application_start_module
            .as_ref()
            .map(|module| tuple("mod", &format!("'{}'", module.replace("/", "@"))))
            .unwrap_or_default();

        let modules = modules
            .iter()
            .map(|m| m.name.replace("/", "@"))
            .sorted()
            .join(",\n               ");

        // TODO: When precompiling for production (i.e. as a precompiled hex
        // package) we will need to exclude the dev deps.
        let applications = config
            .dependencies
            .keys()
            .chain(
                config
                    .dev_dependencies
                    .keys()
                    .take_while(|_| self.config.include_dev_deps),
            )
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            // TODO: test this!
            .map(|name| self.config.package_name_overrides.get(name).unwrap_or(name))
            .chain(config.erlang.extra_applications.iter())
            .sorted()
            .join(",\n                    ");

        let text = format!(
            r#"{{application, {package}, [
{start_module}    {{vsn, "{version}"}},
    {{applications, [{applications}]}},
    {{description, "{description}"}},
    {{modules, [{modules}]}},
    {{registered, []}}
]}}.
"#,
            applications = applications,
            description = config.description,
            modules = modules,
            package = config.name,
            start_module = start_module,
            version = config.version,
        );

        writer.write(&path, &text)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TypeScriptDeclarations {
    None,
    Emit,
}

#[derive(Debug)]
pub struct JavaScript<'a> {
    output_directory: &'a Utf8Path,
    prelude_location: &'a Utf8Path,
    typescript: TypeScriptDeclarations,
}

impl<'a> JavaScript<'a> {
    pub fn new(
        output_directory: &'a Utf8Path,
        typescript: TypeScriptDeclarations,
        prelude_location: &'a Utf8Path,
    ) -> Self {
        Self {
            prelude_location,
            output_directory,
            typescript,
        }
    }

    pub fn render(&self, writer: &impl FileSystemWriter, modules: &[Module]) -> Result<()> {
        for module in modules {
            let js_name = module.name.clone();
            if self.typescript == TypeScriptDeclarations::Emit {
                self.ts_declaration(writer, module, &js_name)?;
            }
            self.js_module(writer, module, &js_name)?
        }
        self.write_prelude(writer)?;
        Ok(())
    }

    fn write_prelude(&self, writer: &impl FileSystemWriter) -> Result<()> {
        let rexport = format!("export * from \"{}\";\n", self.prelude_location);
        writer.write(&self.output_directory.join("gleam.mjs"), &rexport)?;

        if self.typescript == TypeScriptDeclarations::Emit {
            let rexport = rexport.replace(".mjs", ".d.mts");
            writer.write(&self.output_directory.join("gleam.d.mts"), &rexport)?;
        }

        Ok(())
    }

    fn ts_declaration(
        &self,
        writer: &impl FileSystemWriter,
        module: &Module,
        js_name: &str,
    ) -> Result<()> {
        let name = format!("{js_name}.d.mts");
        let path = self.output_directory.join(name);
        let output = javascript::ts_declaration(&module.ast, &module.input_path, &module.code);
        tracing::debug!(name = ?js_name, "Generated TS declaration");
        writer.write(&path, &output?)
    }

    fn js_module(
        &self,
        writer: &impl FileSystemWriter,
        module: &Module,
        js_name: &str,
    ) -> Result<()> {
        let name = format!("{js_name}.mjs");
        let path = self.output_directory.join(name);
        let line_numbers = LineNumbers::new(&module.code);
        let output =
            javascript::module(&module.ast, &line_numbers, &module.input_path, &module.code);
        tracing::debug!(name = ?js_name, "Generated js module");
        writer.write(&path, &output?)
    }
}
