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
use std::{fmt::Debug, fs::File, io::Write, mem, sync::Arc};
use wast::{core::{Expression, Func, FuncKind, FunctionType, InlineExport, Instruction, TypeUse, ValType}, token::{Id, Index, NameAnnotation, Span}, Error};

use crate::analyse::TargetSupport;
use crate::ast::{ArgNames, Assignment, CallArg, Definition, Function, Pattern, Statement, TypedExpr};
use crate::type_::{ModuleInterface, Type};
use camino::Utf8Path;
use wast::core::Instruction::{I64Add, I64Const, I64Sub, LocalGet, LocalSet, StructGet, StructNew};
use wast::core::{HeapType, Local, ModuleField, ModuleKind, RefType, StorageType, StructAccess, StructField, StructType, TypeDef};

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

struct WasmThing {
    gleam_module: crate::ast::Module<ModuleInterface, Definition<Arc<Type>, TypedExpr, EcoString, EcoString>>,
    wasm_instructions: RefCell<Vec<ModuleField<'static>>>,
    //AST
    //Id is pretty private :( identifiers: HashMap<&'a str, Id<'a>>, // Symbol table, but not really, wanted to use for wasm names but unnecessary byte code. Will matter if we do in Gleam  "let x=1; ds(x);"
    identifiers: HashMap<String, usize>, //globals?
    known_types: RefCell<HashMap<&'static str, ValType<'static>>>,
    function_names: HashMap<&'static str,(&'static str, u32)>,
}

fn known_types() -> RefCell<HashMap<&'static str, ValType<'static>>> {
    let mut map = HashMap::new();
    let _ = map.insert("Int", ValType::I64);
    RefCell::new(map)
}

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

    fn transform(mut self) -> std::result::Result<Vec<u8>, Error> {
        let offset = 0; //For now we pretend each is a
                        // let name = self.gleam_module.name;

        // self.names.insert("lol","lol");
        let mut funcidx = 0;
        for definition in &self.gleam_module.definitions {
            // TODO also struct defs
            if let Definition::Function(gleam_function) = definition{
                //lol dump that on the heap... Not sure about this.... TODO check ecostring maybe has some way to do this already?
                let name = gleam_function.name.to_string();
                let name = Box::new(name);
                let name = Box::leak(name);
                let _ = self.function_names.insert(name, (name, funcidx));
                // let _ = self.identifiers.insert(name.to_string(),i); //TODO remove this, will be wrong!
                funcidx = funcidx + 1;
            }

            if let Definition::CustomType(gleam_custom_type)  = definition{
                let name = gleam_custom_type.name.to_string();
                let name = Box::new(name);
                let name = Box::leak(name);
                let _ = self.function_names.insert(name, (name, funcidx));
                // let _ = self.known_types.borrow_mut().insert(name,ValType::Ref(RefType {
                //     nullable: false,
                //     heap: HeapType::Struct,
                // })); //TODO move up? Also nullable but we know its' not... But maybe later

                let _ = self.known_types.borrow_mut().insert(name, ValType::Ref(RefType {
                    nullable: false,
                    heap: HeapType::Concrete(Index::Num(funcidx,Span::from_offset(0))),
                }));

                funcidx = funcidx + 1; //TODO check if same index! Not really but Value constr to be added here? As a func that returns a ref...
                let mut fields = Vec::new();

                for arg in &gleam_custom_type.constructors[0].arguments { //TODO just the first again :(
                    fields.push(StructField {
                        id: None,
                        mutable: false,
                        ty: StorageType::Val(
                            self.transform_gleam_type(arg.type_.as_ref()) //TODO this order will be wrong if we take as argument another type defined later
                        ),
                    })
                }

                self.wasm_instructions.borrow_mut().push(ModuleField::Type(wast::core::Type {
                    span: Span::from_offset(gleam_custom_type.location.start as usize),
                    id:  None,
                    name: None,
                    def: TypeDef::Struct(StructType {
                        fields,
                    }),
                    parent: None,
                    final_type: None,
                }))
            }
        }

        for gleam_definition in &self.gleam_module.definitions {
            self.transform_gleam_definition(gleam_definition);
        }

        let mut wasm_module = wast::core::Module {
            span: Span::from_offset(offset),
            id: None,
            name: None, //Some(NameAnnotation{name: &name}), //Could be none if encoded?
            kind: ModuleKind::Text(self.wasm_instructions.take()),
        };

        wasm_module.encode()
    }

    fn transform_gleam_definition(
        &self,
        gleam_expression: &Definition<Arc<Type>, TypedExpr, EcoString, EcoString>,
    ) {
        match gleam_expression {
            Definition::Function(gleam_function) => {
                self.add_gleam_function_to_wasm_instructions(gleam_function);
            }
            Definition::CustomType(gleam_custom_type) => {
                //TODO extract to method
                //TODO, add to types but how in the AST we have?

                // dbg!(gleam_custom_type);
                // Has constructors field, so add them? TODO dependencies when structs contain structs...
                // Think JVM problems primitives vs. objects, will we get the same problems semantics here?
                // So a constructor would be a function that takes args and returns a structref?
                // let constr = &gleam_custom_type.constructors[0]; //TODO so just one now, but enums! Maybe with tag based on (constructor/variant)name? To match on?
                //Whether ref or not is transparant in Gleam code? No programmer control there...
                // Has name and argument fields..

                //TODO transform to Gleamfunction and use that code?
                let (_,idx) = self.function_names.get(gleam_custom_type.name.as_str()).unwrap();

                let offset = gleam_custom_type.location.start as usize;
                let span = Span::from_offset(offset);

                // let result_type = ValType::Ref(RefType::r#struct()); //TODO aaaah so this is wrong!
                // let result_type = ValType::I64;
                // let result_type = ValType::Ref(RefType {
                //     nullable: false,
                //     heap: HeapType::Struct,
                // });

                let result_type = ValType::Ref(RefType {
                    nullable: false,
                    heap: HeapType::Concrete(Index::Num(*idx,Span::from_offset(0))),
                });

                let mut params: Box<[(Option<Id<'static>>, Option<NameAnnotation<'static>>, ValType<'static>)]> =
                    Box::new([]);
                let mut arguments = Vec::from(mem::take(&mut params));
                let mut locals_box: Box<[Local<'static>]> = Box::new([]);
                let mut locals = Vec::from(mem::take(&mut locals_box)); //TODO why not just the vec?
                let mut instrs: Box<[Instruction<'static>]> = Box::new([]);
                let mut instructions = Vec::from(mem::take(&mut instrs));
                for (i,param) in gleam_custom_type.constructors[0].arguments.iter().enumerate() { //TODO just 1 :(
                    // Put args on stack then, use with stuct new?
                    instructions.push(LocalGet(Index::Num(i as u32,Span::from_offset(0)))); //TODO span
                    let type_ = self.transform_gleam_type(param.type_.as_ref());
                    arguments.push((None, None, type_));
                }

                let creation = StructNew(Index::Num(*idx,Span::from_offset(0))); //TODO span
                instructions.push(creation);

                // instructions.push(LocalGet(Index::Num(0,Span::from_offset(0)))); //TODO remove but hoped struct.new would get me the struct ref :P And ehm so the docs seem to suggest it does push that to the stack p.120: https://webassembly.github.io/gc/core/_download/WebAssembly.pdf


                let ty = TypeUse {
                    index: None,
                    inline: Some(FunctionType {
                        params: arguments.into(),
                        results: Box::new([result_type]),
                    }),
                };

                let export: InlineExport<'_> = if gleam_custom_type.public {
                    // We can have a parser? Inline::parse(MyParser<'a>) That has a &'a to a ParseBuf<'a> which has a from str... beh but takes its lifetime
                    InlineExport {
                        names: vec![self.function_names.get(gleam_custom_type.name.as_str()).unwrap().0] //TODO borrow doesn't live long enough... Can't borrow function for 'a since in a for loop... Like the underlying thing lives long enough Or can I supply a &'a something here?
                    }
                }
                else {
                    InlineExport::default()
                };

                let wasm_func = Func {
                    span,
                    id: None,
                    name: None, //Some(NameAnnotation { name: &gleam_function.name }),
                    exports: export,
                    kind: FuncKind::Inline {
                        locals: locals.into(), //TODO maybe get from scope, it'd bet the slice of it that's bigger than the arguments length...
                        expression: Expression {
                            instrs: instructions.into(),
                        },
                    },
                    ty,
                };
                self.wasm_instructions
                    .borrow_mut()
                    .push(ModuleField::Func(wasm_func));


            },
            _ => todo!()
        }
    }

    fn add_gleam_function_to_wasm_instructions(
        &self,
        gleam_function: &Function<Arc<Type>, TypedExpr>,
    ) {
        let offset = gleam_function.location.start as usize;
        let span = Span::from_offset(offset);
        let result_type = self.transform_gleam_type(gleam_function.return_type.as_ref());
        let mut params: Box<[(Option<Id<'static>>, Option<NameAnnotation<'static>>, ValType<'static>)]> =
            Box::new([]);
        let mut arguments = Vec::from(mem::take(&mut params));
        let mut locals_box: Box<[Local<'static>]> = Box::new([]);
        let mut locals = Vec::from(mem::take(&mut locals_box)); //TODO why not just the vec?
        let mut scope = self.identifiers.clone(); //TODO identifiers is more globals? Not mutated right now..
        for param in gleam_function.arguments.iter() {
            let name = self.get_gleam_name(&param.names);
            let _ = scope.insert(name, scope.len());
            let type_ = self.transform_gleam_type(param.type_.as_ref());
            arguments.push((None, None, type_));
        }
        let mut instrs: Box<[Instruction<'static>]> = Box::new([]);
        let mut instructions = Vec::from(mem::take(&mut instrs));
        for gleam_statement in gleam_function.body.iter() {
            let (mut instrs, mut lcls) = self.transform_gleam_statement(gleam_statement, &mut scope);
            instructions.append(&mut instrs);
            locals.append(&mut lcls);
        }

        let ty = TypeUse {
            index: None,
            inline: Some(FunctionType {
                params: arguments.into(),
                results: Box::new([result_type]),
            }),
        };


        let export: InlineExport<'_> = if gleam_function.public {
            // We can have a parser? Inline::parse(MyParser<'a>) That has a &'a to a ParseBuf<'a> which has a from str... beh but takes its lifetime
            InlineExport {
                names: vec![self.function_names.get(gleam_function.name.as_str()).unwrap().0] //TODO borrow doesn't live long enough... Can't borrow function for 'a since in a for loop... Like the underlying thing lives long enough Or can I supply a &'a something here?
            }
        }
            else {
                InlineExport::default()
            };

        let wasm_func = Func {
            span,
            id: None,
            name: None, //Some(NameAnnotation { name: &gleam_function.name }),
            exports: export,
            kind: FuncKind::Inline {
                locals: locals.into(), //TODO maybe get from scope, it'd bet the slice of it that's bigger than the arguments length...
                expression: Expression {
                    instrs: instructions.into(),
                },
            },
            ty,
        };
        self.wasm_instructions
            .borrow_mut()
            .push(ModuleField::Func(wasm_func));
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
        scope: &mut HashMap<String, usize>,
    ) -> (Vec<Instruction<'static>>, Vec<Local<'static>>) {
        match gleam_statement {
            Statement::Expression(gleam_expression) => {
                self.transform_gleam_expression(gleam_expression, scope)
            },
            Statement::Assignment(gleam_assignment) => {
                self.transform_gleam_assignment(gleam_assignment, scope)
            }
            _ => todo!(),
        }
    }

    fn transform_gleam_assignment(&self, gleam_assignment: &Assignment<Arc<Type>, TypedExpr>, scope: &mut HashMap<String, usize>) -> (Vec<Instruction<'static>>, Vec<Local<'static>>) {
        match &gleam_assignment.pattern {
            Pattern::Variable { name, type_,location } => {
                let idx = scope.len();
                let _ = scope.insert(name.to_string(),scope.len());
                let locals = vec![Local {
                    id: None,
                    name: None,
                    ty: self.transform_gleam_type(type_),
                }];
                let mut instrs = Vec::new();
                let mut val = self.transform_gleam_expression(gleam_assignment.value.as_ref(), scope);
                instrs.append(&mut val.0);
                instrs.push( LocalSet(Index::Num(idx as u32, Span::from_offset(location.start as usize))));
                (instrs,locals)
            },
            _ => todo!()
        }
    }

    fn transform_gleam_expression(
        &self,
        gleam_expression: &TypedExpr,
        scope: &mut HashMap<String, usize>,
    ) -> (Vec<Instruction<'static>>, Vec<Local<'static>>) {
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
            TypedExpr::Var { name, location, .. } => {
                let idx = scope
                    .get(name.as_str())
                    .expect("I expect all vars to be in the scope right now."); //TODO globals different... Need some logic here to decide the local/global get if necessary
                return (vec![Instruction::LocalGet(Index::Num(
                    *idx as u32,
                    Span::from_offset(location.start as usize),
                ))],vec![]);
            },
            TypedExpr::Int{  value, .. } => {
                //TODO type?
               return (vec![I64Const(value.parse().unwrap())],vec![]);
            },
            TypedExpr::Call { location, fun, args, .. } => {
                let mut instrs = Vec::with_capacity(args.len() + 1);
                let mut locals = Vec::new();
                for CallArg{value, ..} in args {
                    // TODO Or this after call?
                    // let mut new_scope = HashMap::new(); panics hehe, vars not in scope..
                   let (mut is, mut ls) =  self.transform_gleam_expression(value, scope);
                    instrs.append(&mut is);
                    locals.append(&mut ls);
                }

                let fn_name = if let TypedExpr::Var{name, .. } = fun.as_ref() {
                    //TODO the start end is stupid, besides Var has more info that gets to fn name directly, also it's the loc of the call not the func
                    // self.start_end_names.get(&(location.start,location.end)).unwrap()
                    name
                } else {
                    dbg!(&fun);
                    todo!()
                };

                // let fn_name = if let &TypedExpr::Fn{location, .. } = fun.as_ref() {
                //     self.start_end_names.get(&(location.start,location.end)).unwrap()
                // } else {
                //     dbg!(&fun);
                //     todo!()
                // };
                // let fn_idx = self.names.get(fn_name).unwrap(); //TODO unwrap, maybe trust Gleam AST but check
                let fn_idx = self.function_names.get(fn_name.as_str()).unwrap().1; //TODO unwrap, maybe trust Gleam AST but check
                let call = Instruction::Call {   //TODO tail call use instead? CallReturn :)
                    0: Index::Num(fn_idx,Span::from_offset(location.start as usize))
                    // 0: Index::Id(Id::new(*fn_idx, Span::from_offset(location.start as usize))), //TODO ugh new is private
                };
                instrs.push(call);
                return (instrs,locals)
            },
            TypedExpr::RecordAccess{ location, typ, label, index, record }  => {
                //TODO so uhm I'd expect stuff before..... oh but take it from the record? Because looked like there was a call to a val constructor but I don't have that at all yet...
                //TODO what is s vs. u in get on wasm side?
                // dbg!("whut");
                // dbg!(record);
                // dbg!(record.type_().named_type_name().unwrap().1);
                // dbg!(label); // "name"
                // dbg!(&typ.named_type_name().unwrap()); //Dangit this is module, int tuple.

                //todo put struct on stack?
                // Ok this is the real problem now, need the ref on the stack before access, duh but do we have that info?

                let record_name = match record.as_ref() {
                    TypedExpr::Var{ name, .. } => {
                        name
                    },
                    _ => {todo!()}
                };

                let r_idx = *scope.get(record_name.as_str()).unwrap() as u32;
                let local_get = LocalGet(Index::Num(r_idx,Span::from_offset(0))); //TODO dang I know it's the third with my example but how to generalize?
                //I do have the scope.. Do I have the var name? cat1 in example
                // dbg!(&location, &typ, &label, &index, &record);


                let struct_idx = self.function_names.get(record.type_().named_type_name().unwrap().1.as_str()).unwrap().1;
                // let struct_idx = 12;
                let struct_get = StructGet(StructAccess{
                    r#struct: Index::Num(struct_idx,Span::from_offset(0)),//TODO offset!
                    field: Index::Num(*index as u32, Span::from_offset(0)), //TODO offset!
                });
                // todo!()
                return (vec![local_get,struct_get],vec![]);
            },
            x => {dbg!(x);todo!()},
        }
        (instructions,locals)
    }

    fn transform_gleam_bin_op(&self, name: &BinOp) -> Instruction<'static> {
        match name {
            BinOp::AddInt => I64Add,
            BinOp::SubInt => I64Sub,
            _ => todo!(),
        }
    }

    fn transform_gleam_type(&self, type_: &Type) -> ValType<'static> {
        match type_ {
            Type::Named { name, .. } => self
                .known_types
                .borrow()
                .get(name.as_str())
                .expect("For now we expect to know all types")
                .clone(),
            _ => todo!(),
        }
    }
}

#[test]
fn wasm_2n() {
    // use wast::core::{Module, ModuleKind,ModuleField};

    let gleam_module = trying_to_make_module(
        "fn add(x: Int, y: Int) -> Int {
            x + y
          }",
    ); //TODO small change removed pub from fn! Since not exported in wasm yet.

    let w = WasmThing {
        gleam_module,
        wasm_instructions: RefCell::new(vec![]),
        identifiers: Default::default(),
        known_types: known_types(), // TODO prolly need types imported and a whole thing when getting some more
        function_names: HashMap::new(),
    };
    let res = w.transform().unwrap();
    let mut file = File::create("letstry.wasm").unwrap();

    let _ = file.write_all(&res);
    // assert!(false);
}


#[test]
fn wasm_3nd() {
    // use wast::core::{Module, ModuleKind,ModuleField};

    let gleam_module = trying_to_make_module(
        "pub fn add(x: Int, y: Int) -> Int {
            let z = 10
            let a = 100
            x + y + z + a
          }",
    );

    let w = WasmThing {
        gleam_module,
        wasm_instructions: RefCell::new(vec![]),
        identifiers: Default::default(),
        known_types: known_types(), // TODO prolly need types imported and a whole thing when getting some more
        function_names: HashMap::new(),
    };
    let res = w.transform().unwrap();
    let mut file = File::create("letstry.wasm").unwrap();

    let _ = file.write_all(&res);
    // assert!(false);
}

#[test]
fn wasm_4nd() {
    let gleam_module = trying_to_make_module(
        "
        pub fn add(x: Int, y: Int) -> Int {
            internal_add(x,y)
          }
        fn internal_add(x: Int, y: Int) -> Int {
            x + y
        }
          ",
    );

    let w = WasmThing {
        gleam_module,
        wasm_instructions: RefCell::new(vec![]),
        identifiers: Default::default(),
        known_types: known_types(), // TODO prolly need types imported and a whole thing when getting some more
        function_names: HashMap::new(),
    };
    let res = w.transform().unwrap();
    let mut file = File::create("letstry.wasm").unwrap();

    let _ = file.write_all(&res);
    // assert!(false);
}

#[test]
fn wasm_5nd() {
//TODO pub types!
    let gleam_module = trying_to_make_module(
        "
         type Cat {
  Cat(name: Int, cuteness: Int)
}
        pub fn add(x: Int, y: Int) -> Int {
            let cat1 = Cat(name: x, cuteness: y)
            cat1.cuteness + cat1.name
          }",
    );


    //TODO: Uncaught (in promise) CompileError: wasm validation error: at offset 43: type mismatch: expression has type i64 but expected structref
    // but we do get bytes.... not promising! Since encode doesn't catch it....
    // Ok new error: CompileError: wasm validation error: at offset 46: not a struct type
    // Yeah cause function does it's own magic (on wasm tools side) to add to the types at top of module, but struct not so much...
    // Ok now: CompileError: wasm validation error: at offset 68: popping value from empty stack
    // Lol was using firefox, maybe no GC? Chrome has better errors: WebAssembly.instantiateStreaming(): Compiling function #1 failed: not enough arguments on the stack for struct.get (need 1, got 0) @+68
// Can also do: wasm2wat -v --enable-gc compiler-core/letstry.wasm
    // now error (Chrome) is: Compiling function #1 failed: struct.get[0] expected type (ref null 0), found local.get of type structref @+68
//Ok sure firefox supports it too

    // wasm2wat still thinks it's wrong even with --enable-all: 0000017: error: expected valid result type (got -0x1c)

    // dbg!(&gleam_module);
    // assert!(false);
    //TODO what the cat type looks like is in module.types

    let w = WasmThing {
        gleam_module,
        wasm_instructions: RefCell::new(vec![]),
        identifiers: Default::default(),
        known_types: known_types(), // TODO prolly need types imported and a whole thing when getting some more
        function_names: HashMap::new(),
    };
    let res = w.transform().unwrap();
    let mut file = File::create("letstry.wasm").unwrap();

    let _ = file.write_all(&res);
    // assert!(false);
}

#[test]
fn wasm_6nd() {
//TODO pub types!
    let gleam_module = trying_to_make_module(
        "
         type Cat {
  Cat(name: Int, cuteness: Int)
}

type Kitten {Kitten(name: Int, age: Int, cuteness: Int) }

        fn add_cat(cat: Cat) -> Int {
    cat.cuteness + cat.name
}

    fn grow(kitten: Kitten) -> Cat {
    Cat(name: kitten.name, cuteness: kitten.cuteness-1)
}

        pub fn add(x: Int, y: Int) -> Int {
            let kitten = Kitten(name: x, cuteness: y, age: 12)
            let cat = grow(kitten)
            add_cat(cat)
          }",
    );

    let w = WasmThing {
        gleam_module,
        wasm_instructions: RefCell::new(vec![]),
        identifiers: Default::default(),
        known_types: known_types(), // TODO prolly need types imported and a whole thing when getting some more
        function_names: HashMap::new(),
    };
    let res = w.transform().unwrap();
    let mut file = File::create("letstry.wasm").unwrap();

    let _ = file.write_all(&res);
    // assert!(false);
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
