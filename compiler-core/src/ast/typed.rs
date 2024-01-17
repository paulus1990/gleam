use std::sync::OnceLock;

use super::*;
use crate::type_::{bool, HasType, Type};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypedExpr {
    Int {
        location: SrcSpan,
        typ: Arc<Type>,
        value: EcoString,
    },

    Float {
        location: SrcSpan,
        typ: Arc<Type>,
        value: EcoString,
    },

    String {
        location: SrcSpan,
        typ: Arc<Type>,
        value: EcoString,
    },

    Block {
        location: SrcSpan,
        statements: Vec1<TypedStatement>,
    },

    /// A chain of pipe expressions.
    /// By this point the type checker has expanded it into a series of
    /// assignments and function calls, but we still have a Pipeline AST node as
    /// even though it is identical to `Block` we want to use different
    /// locations when showing it in error messages, etc.
    Pipeline {
        location: SrcSpan,
        assignments: Vec<TypedAssignment>,
        finally: Box<Self>,
    },

    Var {
        location: SrcSpan,
        constructor: ValueConstructor,
        name: EcoString,
    },

    Fn {
        location: SrcSpan,
        typ: Arc<Type>,
        is_capture: bool,
        args: Vec<Arg<Arc<Type>>>,
        body: Vec1<Statement<Arc<Type>, Self>>,
        return_annotation: Option<TypeAst>,
    },

    List {
        location: SrcSpan,
        typ: Arc<Type>,
        elements: Vec<Self>,
        tail: Option<Box<Self>>,
    },

    Call {
        location: SrcSpan,
        typ: Arc<Type>,
        fun: Box<Self>,
        args: Vec<CallArg<Self>>,
    },

    BinOp {
        location: SrcSpan,
        typ: Arc<Type>,
        name: BinOp,
        left: Box<Self>,
        right: Box<Self>,
    },

    Case {
        location: SrcSpan,
        typ: Arc<Type>,
        subjects: Vec<Self>,
        clauses: Vec<Clause<Self, Arc<Type>, EcoString>>,
    },

    RecordAccess {
        location: SrcSpan,
        typ: Arc<Type>,
        label: EcoString,
        index: u64,
        record: Box<Self>,
    },

    ModuleSelect {
        location: SrcSpan,
        typ: Arc<Type>,
        label: EcoString,
        module_name: EcoString,
        module_alias: EcoString,
        constructor: ModuleValueConstructor,
    },

    Tuple {
        location: SrcSpan,
        typ: Arc<Type>,
        elems: Vec<Self>,
    },

    TupleIndex {
        location: SrcSpan,
        typ: Arc<Type>,
        index: u64,
        tuple: Box<Self>,
    },

    Todo {
        location: SrcSpan,
        message: Option<EcoString>,
        type_: Arc<Type>,
    },

    Panic {
        location: SrcSpan,
        message: Option<EcoString>,
        type_: Arc<Type>,
    },

    BitArray {
        location: SrcSpan,
        typ: Arc<Type>,
        segments: Vec<TypedExprBitArraySegment>,
    },

    RecordUpdate {
        location: SrcSpan,
        typ: Arc<Type>,
        spread: Box<Self>,
        args: Vec<TypedRecordUpdateArg>,
    },

    NegateBool {
        location: SrcSpan,
        value: Box<Self>,
    },

    NegateInt {
        location: SrcSpan,
        value: Box<Self>,
    },
}

impl TypedExpr {
    // This could be optimised in places to exit early if the first of a series
    // of expressions is after the byte index.
    pub fn find_node(&self, byte_index: u32) -> Option<Located<'_>> {
        match self {
            Self::Var { .. }
            | Self::Int { .. }
            | Self::Todo { .. }
            | Self::Panic { .. }
            | Self::Float { .. }
            | Self::String { .. }
            | Self::ModuleSelect { .. } => self.self_if_contains_location(byte_index),

            Self::Pipeline {
                assignments,
                finally,
                ..
            } => assignments
                .iter()
                .find_map(|e| e.find_node(byte_index))
                .or_else(|| finally.find_node(byte_index)),

            Self::Block { statements, .. } => {
                statements.iter().find_map(|e| e.find_node(byte_index))
            }

            Self::Tuple {
                elems: expressions, ..
            }
            | Self::List {
                elements: expressions,
                ..
            } => expressions
                .iter()
                .find_map(|e| e.find_node(byte_index))
                .or_else(|| self.self_if_contains_location(byte_index)),

            Self::NegateBool { value, .. } | Self::NegateInt { value, .. } => value
                .find_node(byte_index)
                .or_else(|| self.self_if_contains_location(byte_index)),

            Self::Fn { body, args, .. } => args
                .iter()
                .find_map(|arg| arg.find_node(byte_index))
                .or_else(|| body.iter().find_map(|s| s.find_node(byte_index)))
                .or_else(|| self.self_if_contains_location(byte_index)),

            Self::Call { fun, args, .. } => args
                .iter()
                .find_map(|arg| arg.find_node(byte_index))
                .or_else(|| fun.find_node(byte_index))
                .or_else(|| self.self_if_contains_location(byte_index)),

            Self::BinOp { left, right, .. } => left
                .find_node(byte_index)
                .or_else(|| right.find_node(byte_index)),

            Self::Case {
                subjects, clauses, ..
            } => subjects
                .iter()
                .find_map(|subject| subject.find_node(byte_index))
                .or_else(|| clauses.iter().find_map(|c| c.find_node(byte_index)))
                .or_else(|| self.self_if_contains_location(byte_index)),

            Self::RecordAccess {
                record: expression, ..
            }
            | Self::TupleIndex {
                tuple: expression, ..
            } => expression
                .find_node(byte_index)
                .or_else(|| self.self_if_contains_location(byte_index)),

            Self::BitArray { segments, .. } => segments
                .iter()
                .find_map(|arg| arg.find_node(byte_index))
                .or_else(|| self.self_if_contains_location(byte_index)),

            Self::RecordUpdate { spread, args, .. } => args
                .iter()
                .find_map(|arg| arg.find_node(byte_index))
                .or_else(|| spread.find_node(byte_index))
                .or_else(|| self.self_if_contains_location(byte_index)),
        }
    }

    fn self_if_contains_location(&self, byte_index: u32) -> Option<Located<'_>> {
        if self.location().contains(byte_index) {
            Some(self.into())
        } else {
            None
        }
    }

    pub fn non_zero_compile_time_number(&self) -> bool {
        use regex::Regex;
        static NON_ZERO: OnceLock<Regex> = OnceLock::new();

        matches!(
            self,
            Self::Int{ value, .. } | Self::Float { value, .. } if NON_ZERO.get_or_init(||


                Regex::new(r"[1-9]").expect("NON_ZERO regex")).is_match(value)
        )
    }

    pub fn location(&self) -> SrcSpan {
        match self {
            Self::Fn { location, .. }
            | Self::Int { location, .. }
            | Self::Var { location, .. }
            | Self::Todo { location, .. }
            | Self::Case { location, .. }
            | Self::Call { location, .. }
            | Self::List { location, .. }
            | Self::Float { location, .. }
            | Self::BinOp { location, .. }
            | Self::Tuple { location, .. }
            | Self::Panic { location, .. }
            | Self::Block { location, .. }
            | Self::String { location, .. }
            | Self::NegateBool { location, .. }
            | Self::NegateInt { location, .. }
            | Self::Pipeline { location, .. }
            | Self::BitArray { location, .. }
            | Self::TupleIndex { location, .. }
            | Self::ModuleSelect { location, .. }
            | Self::RecordAccess { location, .. }
            | Self::RecordUpdate { location, .. } => *location,
        }
    }

    pub fn type_defining_location(&self) -> SrcSpan {
        match self {
            Self::Fn { location, .. }
            | Self::Int { location, .. }
            | Self::Var { location, .. }
            | Self::Todo { location, .. }
            | Self::Case { location, .. }
            | Self::Call { location, .. }
            | Self::List { location, .. }
            | Self::Float { location, .. }
            | Self::BinOp { location, .. }
            | Self::Tuple { location, .. }
            | Self::String { location, .. }
            | Self::Panic { location, .. }
            | Self::NegateBool { location, .. }
            | Self::NegateInt { location, .. }
            | Self::Pipeline { location, .. }
            | Self::BitArray { location, .. }
            | Self::TupleIndex { location, .. }
            | Self::ModuleSelect { location, .. }
            | Self::RecordAccess { location, .. }
            | Self::RecordUpdate { location, .. } => *location,
            Self::Block { statements, .. } => statements.last().location(),
        }
    }

    pub fn definition_location(&self) -> Option<DefinitionLocation<'_>> {
        match self {
            TypedExpr::Fn { .. }
            | TypedExpr::Int { .. }
            | TypedExpr::List { .. }
            | TypedExpr::Call { .. }
            | TypedExpr::Case { .. }
            | TypedExpr::Todo { .. }
            | TypedExpr::Panic { .. }
            | TypedExpr::BinOp { .. }
            | TypedExpr::Float { .. }
            | TypedExpr::Tuple { .. }
            | TypedExpr::NegateBool { .. }
            | TypedExpr::NegateInt { .. }
            | TypedExpr::String { .. }
            | TypedExpr::Block { .. }
            | TypedExpr::Pipeline { .. }
            | TypedExpr::BitArray { .. }
            | TypedExpr::TupleIndex { .. }
            | TypedExpr::RecordAccess { .. } => None,

            // TODO: test
            // TODO: definition
            TypedExpr::RecordUpdate { .. } => None,

            // TODO: test
            TypedExpr::ModuleSelect {
                module_name,
                constructor,
                ..
            } => Some(DefinitionLocation {
                module: Some(module_name.as_str()),
                span: constructor.location(),
            }),

            // TODO: test
            TypedExpr::Var { constructor, .. } => Some(constructor.definition_location()),
        }
    }

    pub fn type_(&self) -> Arc<Type> {
        match self {
            Self::NegateBool { .. } => bool(),
            Self::NegateInt { value, .. } => value.type_(),
            Self::Var { constructor, .. } => constructor.type_.clone(),
            Self::Fn { typ, .. }
            | Self::Int { typ, .. }
            | Self::Todo { type_: typ, .. }
            | Self::Case { typ, .. }
            | Self::List { typ, .. }
            | Self::Call { typ, .. }
            | Self::Float { typ, .. }
            | Self::Panic { type_: typ, .. }
            | Self::BinOp { typ, .. }
            | Self::Tuple { typ, .. }
            | Self::String { typ, .. }
            | Self::BitArray { typ, .. }
            | Self::TupleIndex { typ, .. }
            | Self::ModuleSelect { typ, .. }
            | Self::RecordAccess { typ, .. }
            | Self::RecordUpdate { typ, .. } => typ.clone(),
            Self::Pipeline { finally, .. } => finally.type_(),
            Self::Block { statements, .. } => statements.last().type_(),
        }
    }

    pub fn is_literal(&self) -> bool {
        matches!(
            self,
            Self::Int { .. }
                | Self::List { .. }
                | Self::Float { .. }
                | Self::Tuple { .. }
                | Self::String { .. }
                | Self::BitArray { .. }
        )
    }

    /// Returns `true` if the typed expr is [`Var`].
    ///
    /// [`Var`]: TypedExpr::Var
    #[must_use]
    pub fn is_var(&self) -> bool {
        matches!(self, Self::Var { .. })
    }

    pub(crate) fn get_documentation(&self) -> Option<&str> {
        match self {
            TypedExpr::Var { constructor, .. } => constructor.get_documentation(),
            TypedExpr::ModuleSelect { constructor, .. } => constructor.get_documentation(),

            TypedExpr::Int { .. }
            | TypedExpr::Float { .. }
            | TypedExpr::String { .. }
            | TypedExpr::Block { .. }
            | TypedExpr::Pipeline { .. }
            | TypedExpr::Fn { .. }
            | TypedExpr::List { .. }
            | TypedExpr::Call { .. }
            | TypedExpr::BinOp { .. }
            | TypedExpr::Case { .. }
            | TypedExpr::Tuple { .. }
            | TypedExpr::TupleIndex { .. }
            | TypedExpr::Todo { .. }
            | TypedExpr::Panic { .. }
            | TypedExpr::BitArray { .. }
            | TypedExpr::RecordUpdate { .. }
            | TypedExpr::RecordAccess { .. }
            | TypedExpr::NegateBool { .. }
            | TypedExpr::NegateInt { .. } => None,
        }
    }

    /// Returns `true` if the typed expr is [`Case`].
    ///
    /// [`Case`]: TypedExpr::Case
    #[must_use]
    pub fn is_case(&self) -> bool {
        matches!(self, Self::Case { .. })
    }

    /// Returns `true` if the typed expr is [`Pipeline`].
    ///
    /// [`Pipeline`]: TypedExpr::Pipeline
    #[must_use]
    pub fn is_pipeline(&self) -> bool {
        matches!(self, Self::Pipeline { .. })
    }

    pub fn to_string(&self) -> String {
        match self {
            TypedExpr::Int {
                location,
                typ,
                value,
            } => value.to_string(),
            TypedExpr::Float {
                location,
                typ,
                value,
            } => value.to_string(),
            TypedExpr::String {
                location,
                typ,
                value,
            } => value.to_string(),
            TypedExpr::Block {
                location,
                statements,
            } => todo!(),
            TypedExpr::Pipeline {
                location,
                assignments,
                finally,
            } => todo!(),
            TypedExpr::Var {
                location,
                constructor,
                name,
            } => name.to_string(),
            TypedExpr::Fn {
                location,
                typ,
                is_capture,
                args,
                body,
                return_annotation,
            } => {
                let arg_str = args
                    .iter()
                    .map(|arg| match &arg.names {
                        ArgNames::Discard { name } => name.to_string(),
                        ArgNames::LabelledDiscard { label, name, .. } => name.to_string(),
                        ArgNames::Named { name } => name.to_string(),
                        ArgNames::NamedLabelled { name, label, .. } => name.to_string(),
                    })
                    .collect::<Vec<String>>()
                    .join(", ");

                let body_str = body
                    .iter()
                    .map(|statement| match statement {
                        Statement::Expression(expr) => expr.to_string(),
                        Statement::Assignment(assign) => todo!(),
                        Statement::Use(_) => todo!(),
                    })
                    .collect::<String>();

                format!("fn({}) {{ {} }}", arg_str, body_str)
            }
            TypedExpr::List {
                location,
                typ,
                elements,
                tail,
            } => {
                let mut result = "[".to_string();

                for (i, element) in elements.iter().enumerate() {
                    if i > 0 {
                        result.push_str(", ");
                    }
                    result.push_str(&element.to_string());
                }

                if let Some(tail_expr) = tail {
                    result.push_str(", ");
                    result.push_str(&tail_expr.to_string());
                }

                result.push_str("]");
                result
            }
            TypedExpr::Call {
                location,
                typ,
                fun,
                args,
            } => {
                let fun_str = fun.to_string();

                let args_str = if !args.is_empty() {
                    let args_str: String = args
                        .iter()
                        .map(|arg| format!("{}", arg.value.to_string()))
                        .collect::<Vec<_>>()
                        .join(", ");
                    format!("({})", args_str)
                } else {
                    //no arguments for function
                    String::from("()")
                };

                format!("{}{}", fun_str, args_str)
            }
            TypedExpr::BinOp {
                location,
                typ,
                name,
                left,
                right,
            } => {
                format!("{} {} {}", left.to_string(), name.name(), right.to_string())
            }
            TypedExpr::Case {
                location,
                typ,
                subjects,
                clauses,
            } => todo!(),
            TypedExpr::RecordAccess {
                location,
                typ,
                label,
                index,
                record,
            } => todo!(),
            TypedExpr::ModuleSelect {
                location,
                typ,
                label,
                module_name,
                module_alias,
                constructor,
            } => {
                //list.reverse
                dbg!(self);
                format!("{}.{}", module_alias, label)
            }
            TypedExpr::Tuple {
                location,
                typ,
                elems,
            } => {
                let mut res = "(".to_string();

                for (i, elem) in elems.iter().enumerate() {
                    if i > 0 {
                        res.push_str(", ");
                    }
                    res.push_str(&format!("{}", elem.to_string()));
                }

                res.push(')');
                res
            }
            TypedExpr::TupleIndex {
                location,
                typ,
                index,
                tuple,
            } => todo!(),
            TypedExpr::Todo {
                location,
                message,
                type_,
            } => todo!(),
            TypedExpr::Panic {
                location,
                message,
                type_,
            } => todo!(),
            TypedExpr::BitArray {
                location,
                typ,
                segments,
            } => todo!(),
            TypedExpr::RecordUpdate {
                location,
                typ,
                spread,
                args,
            } => todo!(),
            TypedExpr::NegateBool { location, value } => value.to_string(),
            TypedExpr::NegateInt { location, value } => value.to_string(),
        }
    }
}

impl<'a> From<&'a TypedExpr> for Located<'a> {
    fn from(value: &'a TypedExpr) -> Self {
        Located::Expression(value)
    }
}

impl HasLocation for TypedExpr {
    fn location(&self) -> SrcSpan {
        self.location()
    }
}

impl HasType for TypedExpr {
    fn type_(&self) -> Arc<Type> {
        self.type_()
    }
}

impl crate::bit_array::GetLiteralValue for TypedExpr {
    fn as_int_literal(&self) -> Option<i64> {
        if let TypedExpr::Int { value: val, .. } = self {
            if let Ok(val) = val.parse::<i64>() {
                return Some(val);
            }
        }
        None
    }
}
