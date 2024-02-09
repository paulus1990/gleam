use crate::{
    ast::BinOp,
    build::{ErlangAppCodegenConfiguration, Module},
    config::PackageConfig,
    erlang,
    io::FileSystemWriter,
    javascript,
    line_numbers::LineNumbers,
    warning::{TypeWarningEmitter, WarningEmitter},
    Result,
};
use ecow::EcoString;
use im::HashMap;
use itertools::Itertools;
use std::cell::RefCell;
use std::{fmt::Debug, fs::File, io::Write, sync::Arc};

use crate::analyse::TargetSupport;
use crate::ast::{ArgNames, Assignment, CallArg, Definition, Function, Pattern, Statement, TypedExpr};
use crate::type_::{ModuleInterface, Type};
use camino::Utf8Path;
use wasabi_leb128::WriteLeb128;
use crate::codegen::WasmInstruction::{Call, I32Add, I32Const, I32Sub, LocalGet, LocalSet};
use crate::codegen::WasmType::{I32, ConcreteRef};
use crate::codegen::WasmTypeSectionEntry::PlaceHolder;

fn encode_unsigned_leb128(x: u32) -> Vec<u8> {
    //TODO maybe wrong :P
    let mut buf = Vec::new();
    let _ = buf.write_leb128(x).unwrap();
    buf
    // if x == 0
    // {
    //     vec![0]
    // } else {
    //     x.to_le_bytes().into_iter().take_while(|x| *x != 0).collect()
    // }
    // let mut things = x.to_le_bytes().to_vec();
    // let huh = things.len() % 7;
    // let mut y = vec![0,huh];
    // y.append(&mut things);
    // y.into_iter().chunks(7).into_iter().enumerate().map();
    //
    // todo!()
}

fn encode_signed_leb128(mut x: i32) -> Vec<u8> {
    let mut buf = Vec::new();
    let _ = buf.write_leb128(x).unwrap();
    buf
    //TODO wrong for sure
    // let mut result = 0;
    // let mut shift = 0;
    // let mut input = x.to_be_bytes().to_vec();
    // loop {
    //     let byte = input.pop().unwrap_or(0);
    //     result |= (byte & 0x7f) << shift;
    //     shift += 7;
    //     if (0x80 & byte) == 0 {
    //         if shift < 32 && (byte & 0x40) != 0 {
    //             return (result | (u8::MAX << shift)).to_le_bytes().to_vec(); //~0 = ~0 then https://en.wikipedia.org/wiki/LEB128
    //         }
    //         return result.to_le_bytes().to_vec();
    //     }
    // }
}

trait Wasmable {
    fn to_wat(&self) -> EcoString;
    fn to_wasm(&self) -> Vec<u8>;
}

#[derive(Clone, Debug)]
enum WasmType {
    I32,
    ConcreteRef(WasmVar),

}

#[derive(Debug)]
enum WasmTypeSectionEntry {
    PlaceHolder,
    Function(WasmFuncDef),
    Struct(WasmStructDef),
}

impl WasmTypeSectionEntry {
    fn public(&self) -> bool {
        match self {
            WasmTypeSectionEntry::Function(f) => f.exported,
            _ => false
        }
    }
}

impl Wasmable for WasmTypeSectionEntry {
    fn to_wat(&self) -> EcoString {
        match self {
            PlaceHolder => { panic!() }
            WasmTypeSectionEntry::Function(x) => { x.to_wat() }
            WasmTypeSectionEntry::Struct(x) => { x.to_wat() }
        }
    }

    fn to_wasm(&self) -> Vec<u8> {
        match self {
            PlaceHolder => { panic!() }
            WasmTypeSectionEntry::Function(x) => { x.to_wasm() }
            WasmTypeSectionEntry::Struct(x) => { x.to_wasm() }
        }
    }
}

#[derive(Clone, Debug)]
struct WasmFuncDef {
    info: WasmVar,
    params: Vec<WasmType>,
    return_type: WasmType,
    exported: bool,
}

impl Wasmable for WasmFuncDef {
    fn to_wat(&self) -> EcoString {
        "".into() //TODO I don't think we need the sections in wat
    }

    fn to_wasm(&self) -> Vec<u8> {
        //TODO prolly refactor lol
        let mut rename_me = Vec::new();
        rename_me.push(0x60); //Func type
        let param_len = self.params.len();
        rename_me.append(&mut encode_unsigned_leb128(param_len as u32));
        for param in &self.params {
            rename_me.append(&mut param.to_wasm());
        }
        let result_len = 1;
        rename_me.push(result_len);
        rename_me.append(&mut self.return_type.to_wasm());
        rename_me
    }
}

#[derive(Debug)]
struct WasmStructDef {
    info: WasmVar,
    fields: Vec<WasmType>,
}

impl Wasmable for WasmStructDef {
    fn to_wat(&self) -> EcoString {
        todo!()
    }

    fn to_wasm(&self) -> Vec<u8> {
        todo!()
    }
}

impl Wasmable for WasmType {
    fn to_wat(&self) -> EcoString {
        match self {
            I32 => "i32".into(),
            ConcreteRef(x) => format!("ref {} ;; {}", x.idx, x.name).into(),
        }
    }

    fn to_wasm(&self) -> Vec<u8> {
        match self {
            I32 => vec![0x7f],
            ConcreteRef(x) => {
                let mut v = vec![0x64];
                v.append(&mut encode_unsigned_leb128(x.idx));
                v
            }
        }
    }
}

#[derive(Clone, Eq, PartialEq, Debug)]
struct WasmVar {
    idx: u32,
    name: EcoString,
}

impl Wasmable for WasmVar {
    fn to_wat(&self) -> EcoString {
        format!("${}", self.name).into()
    }

    fn to_wasm(&self) -> Vec<u8> {
        encode_unsigned_leb128(self.idx)
    }
}

#[derive(Debug)]
struct WasmFunction {
    // info: WasmVar,
    args: Vec<(WasmVar, WasmType)>,
    def: WasmFuncDef,
    body: Vec<WasmInstruction>,
    locals: Vec<(WasmVar, WasmType)>,
}

impl Wasmable for WasmFunction {
    fn to_wat(&self) -> EcoString {
        let export = if self.def.exported { format!(" (export \"{}\")", self.def.info.name) } else { "".to_string() };
        let ret = format!("(result {})", self.def.return_type.to_wat());
        let args = self.args.iter().fold(EcoString::new(), |mut acc, x| {
            acc.push_str(&format!(" (param ${} {})", x.0.name, x.1.to_wat()));
            acc
        });
        let locals = self.locals.iter().fold(
            EcoString::new(),
            |mut acc, (v, type_)| {
                acc.push_str(&mut format!("\n    (local ${} {})", v.name, type_.to_wat()));
                acc
            },
        );
        let body = self.body.iter().map(|x| format!("\n    ({})", x.to_wat())).fold(EcoString::new(), |mut acc, x| {
            acc.push_str(&x);
            acc
        });

        format!("(func ${}{export}{args} {ret}{locals}{body})", self.def.info.name).into()
    }

    fn to_wasm(&self) -> Vec<u8> {
        todo!()
    }
}

#[derive(Debug)]
enum WasmInstruction {
    LocalGet(WasmVar),
    LocalSet(WasmVar),
    // Const(WasmType, WasmVar),
    Call(WasmVar),
    Function(WasmFunction),
    I32Add,
    I32Sub,
    I32Const(i32),
}

impl Wasmable for WasmInstruction {
    fn to_wat(&self) -> EcoString {
        match self {
            LocalGet(x) => { format!("local.get ${}", x.name).into() }
            LocalSet(x) => { format!("local.set ${}", x.name).into() }
            Call(x) => { format!("call ${}", x.name).into() }
            WasmInstruction::Function(x) => { x.to_wat() }
            I32Add => { "i32.add".into() }
            I32Sub => { "i32.sub".into() }
            I32Const(x) => { format!("i32.const {x}").into() }
        }
    }

    fn to_wasm(&self) -> Vec<u8> {
        match self {
            I32Const(x) => {
                let mut acc = vec![0x41];
                acc.append(&mut encode_signed_leb128(*x));
                acc
            }
            WasmInstruction::Function(f) => {
                let mut func = Vec::new();
                let mut local_count = encode_unsigned_leb128(f.locals.len() as u32);
                let mut body: Vec<u8> = f.body.iter().flat_map(|x| x.to_wasm()).collect();
                let mut locals: Vec<u8> = f.locals.iter().flat_map(|(_, t)| {
                    let mut acc = vec![1]; //var type
                    acc.append(t.to_wasm().as_mut());
                    acc
                }
                ).collect();
                let mut func_size = encode_unsigned_leb128((body.len() + 1 + local_count.len() + locals.len()) as u32); //+1 for end should include local count len????
                func.append(&mut func_size);
                func.append(&mut local_count);
                func.append(&mut locals);
                func.append(&mut body);
                func.push(0x0b); //end function
                func
            }
            LocalGet(v) => {
                let mut acc = vec![0x20];
                acc.append(&mut encode_unsigned_leb128(v.idx));
                acc
            }
            LocalSet(v) => {
                let mut acc = vec![0x21];
                acc.append(&mut encode_unsigned_leb128(v.idx));
                acc
            }
            Call(v) => {
                let mut acc = vec![0x10];
                acc.append(&mut encode_unsigned_leb128(v.idx));
                acc
            }
            I32Add => { vec![0x6a] }
            I32Sub => { vec![0x6b] }
        }
    }
}


fn trying_to_make_module(
    program: &str,
) -> crate::ast::Module<ModuleInterface, Definition<Arc<Type>, TypedExpr, EcoString, EcoString>> {
    let parsed = crate::parse::parse_module(program).expect("syntax error");
    let module = parsed.module;
    // println!("{module:?}");
    let ids = crate::uid::UniqueIdGenerator::new();
    let small = ecow::EcoString::from("Welcome");
    let mut hs = HashMap::new();
    let hs2: std::collections::HashMap<EcoString, EcoString> = std::collections::HashMap::new();
    let we = TypeWarningEmitter::new(
        camino::Utf8PathBuf::new(),
        "".into(),
        WarningEmitter::new(Arc::new(crate::warning::VectorWarningEmitterIO::default())),
    );

    let _ = hs.insert(
        crate::type_::PRELUDE_MODULE_NAME.into(),
        crate::type_::build_prelude(&ids),
    );

    let module = crate::analyse::infer_module(
        crate::build::Target::JavaScript,
        &ids,
        module,
        crate::build::Origin::Src,
        &small,
        &hs,
        &we,
        &hs2,
        TargetSupport::NotEnforced,
    )
        .expect("type error?");
    module
}

pub(crate) struct WasmThing {
    pub(crate) gleam_module: crate::ast::Module<ModuleInterface, Definition<Arc<Type>, TypedExpr, EcoString, EcoString>>,
    pub(crate) wasm_instructions: RefCell<Vec<WasmInstruction>>,
    pub(crate) type_section: RefCell<Vec<WasmTypeSectionEntry>>,
    pub(crate) functions_type_section_index: RefCell<HashMap<EcoString, u32>>,
    // pub(crate) wasm_instructions: RefCell<Vec<ModuleField<'static>>>,
    // //AST
    // //Id is pretty private :( identifiers: HashMap<&'a str, Id<'a>>, // Symbol table, but not really, wanted to use for wasm names but unnecessary byte code. Will matter if we do in Gleam  "let x=1; ds(x);"
    // pub(crate) identifiers: HashMap<String, usize>,
    // //globals?
    // pub(crate) known_types: RefCell<HashMap<&'static str, ValType<'static>>>,
    // pub(crate) function_names: HashMap<&'static str, (&'static str, u32)>,
}

// pub(crate) fn known_types() -> RefCell<HashMap<&'static str, ValType<'static>>> {
//     let mut map = HashMap::new();
//     let _ = map.insert("Int", ValType::I32);
//     RefCell::new(map)
// }

impl WasmThing {
    // TODO remember wasm is stack based so arguments before functions :)


    // TODO give <'a> fn new(gleam_module: crate::ast::Module<crate::type_::ModuleInterface, crate::ast::Definition<Arc<crate::type_::Type>, crate::ast::TypedExpr, EcoString, EcoString>>) -> WasmThing<'a> {
    //     WasmThing {
    //         gleam_module,
    //         wasm_instructions: vec![],
    //         identifiers: Default::default(),
    //         known_types: known_types()
    //         // TODO prolly need types imported and a whole thing when getting some more
    //     }
    // }

    pub(crate) fn transform(&self) -> () {
        for gleam_definition in &self.gleam_module.definitions {
            self.transform_gleam_definition(gleam_definition);
        }
    }

    fn transform_gleam_definition(
        &self,
        gleam_expression: &Definition<Arc<Type>, TypedExpr, EcoString, EcoString>,
    ) {
        match gleam_expression {
            Definition::Function(gleam_function) => {
                self.add_gleam_function_to_wasm_instructions(gleam_function);
            }
            _ => todo!()
        }
    }

    fn add_gleam_function_to_wasm_instructions(
        &self,
        gleam_function: &Function<Arc<Type>, TypedExpr>,
    ) {
        let name = gleam_function.name.clone();
        let len = self.type_section.borrow().len();
        let loc: u32 = *self.functions_type_section_index.borrow_mut().get(&name).unwrap_or(&(len as u32));
        let _ = self.functions_type_section_index.borrow_mut().insert(name.clone(), loc);
        let wasm_var = WasmVar { idx: loc, name };

        let result_type = self.transform_gleam_type(gleam_function.return_type.as_ref());
        let mut arguments = Vec::new();
        let mut locals = Vec::new();
        let mut scope: HashMap<EcoString, usize> = HashMap::new();
        for (i, param) in gleam_function.arguments.iter().enumerate() {
            let name = param.names.get_variable_name().unwrap(); //TODO unwrap???
            let _ = scope.insert(name.clone(), scope.len());
            let type_ = self.transform_gleam_type(param.type_.as_ref());
            arguments.push((WasmVar { idx: i as u32, name: name.clone() }, type_));
        }

        let mut instructions = Vec::new();
        for gleam_statement in gleam_function.body.iter() {
            let (mut instrs, mut lcls) = self.transform_gleam_statement(gleam_statement, &mut scope);
            instructions.append(&mut instrs);
            locals.append(&mut lcls);
        }

        let func_def = WasmFuncDef {
            info: wasm_var,
            params: arguments.iter().map(|x| x.1.clone()).collect(),
            return_type: result_type,
            exported: gleam_function.public,
        };
        if loc >= len as u32 {
            //TODO while! Maybe? Thiink more
            self.type_section.borrow_mut().push(PlaceHolder);
        }
        self.type_section.borrow_mut()[loc as usize] = WasmTypeSectionEntry::Function(func_def.clone()); //TODO grow vec if necess
        let wasm_func = WasmInstruction::Function(
            WasmFunction {
                args: arguments,
                def: func_def,
                body: instructions,
                locals: locals,
            }
        );
        self.wasm_instructions
            .borrow_mut()
            .push(wasm_func);
    }

    fn get_gleam_name(&self, names: &ArgNames) -> String {
        match names {
            ArgNames::Named { name } => return name.to_string(),
            _ => todo!(),
        }
    }

    fn transform_gleam_statement(
        &self,
        gleam_statement: &Statement<Arc<Type>, TypedExpr>,
        scope: &mut HashMap<EcoString, usize>,
    ) -> (Vec<WasmInstruction>, Vec<(WasmVar, WasmType)>) {
        match gleam_statement {
            Statement::Expression(gleam_expression) => {
                self.transform_gleam_expression(gleam_expression, scope)
            }
            Statement::Assignment(gleam_assignment) => {
                self.transform_gleam_assignment(gleam_assignment, scope)
            }
            _ => todo!(),
        }
    }

    fn transform_gleam_assignment(&self, gleam_assignment: &Assignment<Arc<Type>, TypedExpr>, scope: &mut HashMap<EcoString, usize>) -> (Vec<WasmInstruction>, Vec<(WasmVar, WasmType)>) {
        match &gleam_assignment.pattern {
            Pattern::Variable { name, type_, .. } => {
                let idx = scope.len();
                let _ = scope.insert(name.clone(), scope.len());
                let locals = vec![(
                    WasmVar { idx: idx as u32, name: name.clone() }, self.transform_gleam_type(type_),
                )];
                let mut instrs = Vec::new();
                let mut val = self.transform_gleam_expression(gleam_assignment.value.as_ref(), scope);
                instrs.append(&mut val.0);
                instrs.push(LocalSet(locals[0].0.clone()));
                (instrs, locals)
            }
            _ => todo!()
        }
    }

    fn transform_gleam_expression(
        &self,
        gleam_expression: &TypedExpr,
        scope: &mut HashMap<EcoString, usize>,
    ) -> (Vec<WasmInstruction>, Vec<(WasmVar, WasmType)>) {
        let mut instructions = Vec::new();
        let mut locals = Vec::new();
        match gleam_expression {
            TypedExpr::BinOp {
                name, left, right, ..
            } => {
                let mut ls = self.transform_gleam_expression(left.as_ref(), scope);
                instructions.append(&mut ls.0);
                locals.append(&mut ls.1);
                let mut rs = self.transform_gleam_expression(right.as_ref(), scope);
                instructions.append(&mut rs.0);
                instructions.push(self.transform_gleam_bin_op(name));
                locals.append(&mut rs.1);
            }
            TypedExpr::Var { name, .. } => {
                let idx = scope
                    .get(name.as_str())
                    .expect("I expect all vars to be in the scope right now."); //TODO globals different... Need some logic here to decide the local/global get if necessary
                return (vec![LocalGet(WasmVar { idx: *idx as u32, name: name.clone() })], vec![]);
            }
            TypedExpr::Int { value, .. } => {
                //TODO type?
                return (vec![I32Const(value.parse().unwrap())], vec![]);
            }
            TypedExpr::Call { fun, args, .. } => {
                let mut instrs = Vec::with_capacity(args.len() + 1);
                let mut locals = Vec::new();
                for CallArg { value, .. } in args {
                    // TODO Or this after call?
                    // let mut new_scope = HashMap::new(); panics hehe, vars not in scope..
                    let (mut is, mut ls) = self.transform_gleam_expression(value, scope);
                    instrs.append(&mut is);
                    locals.append(&mut ls);
                }

                let fn_name = if let TypedExpr::Var { name, .. } = fun.as_ref() {
                    //TODO the start end is stupid, besides Var has more info that gets to fn name directly, also it's the loc of the call not the func
                    // self.start_end_names.get(&(location.start,location.end)).unwrap()
                    name
                } else {
                    dbg!(&fun);
                    todo!()
                };

                let pi = *self.functions_type_section_index.borrow().get(fn_name.as_str()).unwrap_or(&u32::MAX);
                let fn_idx = match pi {
                    u32::MAX => {
                        let len = self.type_section.borrow().len();
                        self.type_section.borrow_mut().push(PlaceHolder);
                        let _ = self.functions_type_section_index.borrow_mut().insert(fn_name.clone(), len as u32);
                        len as u32
                    }
                    i => { i }
                };
                let call = Call(   //TODO tail call use instead? CallReturn :)
                                   WasmVar {
                                       idx: fn_idx,
                                       name: fn_name.clone(),
                                   }
                );
                instrs.push(call);
                return (instrs, locals);
            }
            // TypedExpr::RecordAccess { index, record, .. } => { TODO
            x => {
                dbg!(x);
                todo!()
            }
        }
        (instructions, locals)
    }

    fn transform_gleam_bin_op(&self, name: &BinOp) -> WasmInstruction {
        match name {
            BinOp::AddInt => I32Add,
            BinOp::SubInt => I32Sub,
            _ => todo!(),
        }
    }

    fn transform_gleam_type(&self, type_: &Type) -> WasmType {
        match type_ {
            Type::Named { name, .. } =>
                match name.as_str() {
                    "Int" => I32,
                    _ => todo!()
                }
            _ => todo!() //Prolly a ref, with correct index?,
        }
    }
}

impl Wasmable for WasmThing {
    fn to_wat(&self) -> EcoString {
        // not necess I guess... Maybe for structs.., TODO!
        // let types = self.type_section.borrow().iter().map(|x| x.to_wat())
        //     .reduce(|mut acc, x| {acc.push_str("\n");acc.push_str(&x); acc}).unwrap();

        let instructions = self.wasm_instructions.borrow().iter().map(|x| x.to_wat())
            .reduce(|mut acc, x| {
                acc.push_str("\n");
                acc.push_str(&x);
                acc
            }).unwrap_or_default();

        let mut module = EcoString::from("(module\n");
        module.push_str(&instructions);
        module.push_str(")");
        module
    }

    fn to_wasm(&self) -> Vec<u8> {
        let mut module = vec![0, 'a' as u8, 's' as u8, 'm' as u8, 1, 0, 0, 0];

        //Type section
        module.push(1);
        let mut section: Vec<u8> = self.type_section.borrow().iter().flat_map(
            |x| x.to_wasm()
        ).collect();
        let entry_count = &mut encode_unsigned_leb128(self.type_section.borrow().len() as u32);
        module.append(&mut encode_unsigned_leb128(section.len() as u32 + entry_count.len() as u32));
        module.append(entry_count);
        module.append(&mut section);

        //Function section
        module.push(3);
        let mut section: Vec<u8> = self.functions_type_section_index.borrow().iter().flat_map(
            |(_, x)| encode_unsigned_leb128(*x)
        ).collect();
        let entry_count = &mut encode_unsigned_leb128(self.functions_type_section_index.borrow().len() as u32);
        module.append(&mut encode_unsigned_leb128(section.len() as u32 + entry_count.len() as u32)); //TODO add bytes of below!
        module.append(entry_count);
        module.append(&mut section);

        //Export section
        module.push(7);

        let mut section: Vec<u8> = self.type_section.borrow().iter().filter(|x| x.public()).flat_map(
            |x| {
                match x {
                    WasmTypeSectionEntry::Function(f) => {
                        let mut fn_bytes = Vec::new();
                        let mut name_bytes = f.info.name.to_ascii_lowercase().as_bytes().to_vec();//nm name TODO lower case? Ascii?
                        fn_bytes.append(&mut encode_unsigned_leb128(name_bytes.len() as u32));
                        fn_bytes.append(&mut name_bytes);
                        fn_bytes.push(0); //function
                        fn_bytes.append(&mut encode_unsigned_leb128(f.info.idx));
                        fn_bytes
                    }
                    _ => { panic!() }
                }
            }
        ).collect();
        let entry_count = &mut encode_unsigned_leb128(self.type_section.borrow().iter().filter(|x| x.public()).count() as u32);
        module.append(&mut encode_unsigned_leb128(section.len() as u32 + entry_count.len() as u32));
        module.append(entry_count);
        module.append(&mut section);

        //Code section
        module.push(0x0a);
        let instr_len = self.wasm_instructions.borrow().len();
        let mut section: Vec<u8> = self.wasm_instructions.borrow().iter().flat_map(
            |instr| instr.to_wasm()
        ).collect();
        let byte_len = section.len() + encode_unsigned_leb128(instr_len as u32).len();
        module.append(&mut encode_unsigned_leb128(byte_len as u32));
        module.append(&mut encode_unsigned_leb128(instr_len as u32));
        module.append(&mut section);

        module
    }
}

#[test]
fn wasm_2n() {
    let gleam_module = trying_to_make_module(
        "pub fn add(x: Int, y: Int) -> Int {
            x + y
          }",
    );

    let w = WasmThing {
        gleam_module,
        wasm_instructions: RefCell::new(vec![]),
        type_section: RefCell::new(vec![]),
        functions_type_section_index: RefCell::new(Default::default()),
    };
    w.transform();
    let wasm = w.to_wasm();
    let wasm_string_bytes = wasm.iter().map(|x| format!("{:#04X?}", *x)).reduce(
        |mut acc, x| {
            acc.push_str("\n");
            acc.push_str(&x);
            acc
        }
    ).unwrap();

    let wat = w.to_wat();
    let mut file = File::create("letstry.wat").unwrap();
    let _ = file.write_all(wat.as_bytes());
    insta::assert_snapshot!(wat);

    let mut file = File::create("letstry.wasm").unwrap();
    let _ = file.write_all(&wasm);
    insta::assert_snapshot!(wasm_string_bytes);
}


#[test]
fn wasm_3nd() {
    let gleam_module = trying_to_make_module(
        "pub fn add(x: Int, y: Int) -> Int {
            let z = 10
            let a = 100
            x+ y + z + a
          }",
    );

    let w = WasmThing {
        gleam_module,
        wasm_instructions: RefCell::new(vec![]),
        type_section: RefCell::new(vec![]),
        functions_type_section_index: RefCell::new(Default::default()),
    };
    w.transform();
    let wasm = w.to_wasm();
    let wasm_string_bytes = wasm.iter().map(|x| format!("{:#04X?}", *x)).reduce(
        |mut acc, x| {
            acc.push_str("\n");
            acc.push_str(&x);
            acc
        }
    ).unwrap();

    let wat = w.to_wat();
    let mut file = File::create("letstry.wat").unwrap();
    let _ = file.write_all(wat.as_bytes());
    insta::assert_snapshot!(wat);

    let mut file = File::create("letstry.wasm").unwrap();
    let _ = file.write_all(&wasm);
    insta::assert_snapshot!(wasm_string_bytes);
}

// #[test]
// fn wasm_4nd() {
//     let gleam_module = trying_to_make_module(
//         "
//         pub fn add(x: Int, y: Int) -> Int {
//             internal_add(x,y)
//           }
//         fn internal_add(x: Int, y: Int) -> Int {
//             x + y
//         }
//           ",
//     );
//
//     let w = WasmThing {
//         gleam_module,
//         wasm_instructions: RefCell::new(vec![]),
//         identifiers: Default::default(),
//         known_types: known_types(), // TODO prolly need types imported and a whole thing when getting some more
//         function_names: HashMap::new(),
//     };
//     let res = w.transform().unwrap();
//     let mut file = File::create("letstry.wasm").unwrap();
//
//     let _ = file.write_all(&res);
//     // assert!(false);
// }
//
// #[test]
// fn wasm_5nd() {
// //TODO pub types!
//     let gleam_module = trying_to_make_module(
//         "
//          type Cat {
//   Cat(name: Int, cuteness: Int)
// }
//         pub fn add(x: Int, y: Int) -> Int {
//             let cat1 = Cat(name: x, cuteness: y)
//             cat1.cuteness + cat1.name
//           }",
//     );
//
//
//     //TODO: Uncaught (in promise) CompileError: wasm validation error: at offset 43: type mismatch: expression has type i64 but expected structref
//     // but we do get bytes.... not promising! Since encode doesn't catch it....
//     // Ok new error: CompileError: wasm validation error: at offset 46: not a struct type
//     // Yeah cause function does it's own magic (on wasm tools side) to add to the types at top of module, but struct not so much...
//     // Ok now: CompileError: wasm validation error: at offset 68: popping value from empty stack
//     // Lol was using firefox, maybe no GC? Chrome has better errors: WebAssembly.instantiateStreaming(): Compiling function #1 failed: not enough arguments on the stack for struct.get (need 1, got 0) @+68
// // Can also do: wasm2wat -v --enable-gc compiler-core/letstry.wasm
//     // now error (Chrome) is: Compiling function #1 failed: struct.get[0] expected type (ref null 0), found local.get of type structref @+68
// //Ok sure firefox supports it too
//
//     // wasm2wat still thinks it's wrong even with --enable-all: 0000017: error: expected valid result type (got -0x1c)
//     // And the problem is the "0x64 0x00" return type of the constructor... parsed as -0x1c, checked by changing that 0x64 byte, will change te -0x1c error msg, wild!
//     // ex: change to 0x63 will say: "000001f: error: expected valid result type (got -0x1d)"
//     // browsers still parse em...
//     // And spec says it's allowed in return type?
//     // check issues: maybe https://github.com/WebAssembly/wabt/issues/2364 (see also: https://github.com/WebAssembly/wabt/pull/2363)? Lol is ref encoded as 0x6b instead of 0x64, that's be nice haha
//     // or https://github.com/WebAssembly/wabt/issues/2333 weird! Has the enable flag but no support?
//     // also --enable-gc does allow it to process the 0x5f struct type flag..
//     // prolly problem here: https://github.com/WebAssembly/wabt/blob/main/include/wabt/type.h#L47 Oh wel..
//     // Eh no compiled locally with line 47 changed the problem is bigger. Also if I change to 6b in file won't fix with original.
//     // Aaah enable more features: /home/harm/git/wabt/build/wasm2wat --enable-all -v /home/harm/git/gleam/compiler-core/letstry.wasm
//     // new error: 0000056: error: unexpected opcode: 0xfb
//     // Ah crap that's struct.new, well if it's not supported it really is not supported....
//
//     //TODO so the concrete types are fine now, but would like abstract struct when returning an enum variant, then you'd need structref (well non-nullable right, I mean...)
//
//     // dbg!(&gleam_module);
//     // assert!(false);
//     //TODO what the cat type looks like is in module.types
//
//     let w = WasmThing {
//         gleam_module,
//         wasm_instructions: RefCell::new(vec![]),
//         identifiers: Default::default(),
//         known_types: known_types(), // TODO prolly need types imported and a whole thing when getting some more
//         function_names: HashMap::new(),
//     };
//     let res = w.transform().unwrap();
//     let mut file = File::create("letstry.wasm").unwrap();
//
//     let _ = file.write_all(&res);
//     // assert!(false);
// }
//
// #[test]
// fn wasm_6nd() {
// //TODO pub types!
//     let gleam_module = trying_to_make_module(
//         "
//          type Cat {
//   Cat(name: Int, cuteness: Int)
// }
//
// type Kitten {Kitten(name: Int, age: Int, cuteness: Int) }
//
//         fn add_cat(cat: Cat) -> Int {
//     cat.cuteness + cat.name
// }
//
//     fn grow(kitten: Kitten) -> Cat {
//     Cat(name: kitten.name, cuteness: kitten.cuteness-1)
// }
//
//         pub fn add(x: Int, y: Int) -> Int {
//             let kitten = Kitten(name: x, cuteness: y, age: 12)
//             let cat = grow(kitten)
//             add_cat(cat)
//           }",
//     );
//
//     let w = WasmThing {
//         gleam_module,
//         wasm_instructions: RefCell::new(vec![]),
//         identifiers: Default::default(),
//         known_types: known_types(), // TODO prolly need types imported and a whole thing when getting some more
//         function_names: HashMap::new(),
//     };
//     let res = w.transform().unwrap();
//     let mut file = File::create("letstry.wasm").unwrap();
//
//     let _ = file.write_all(&res);
//     // assert!(false);
// }

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
