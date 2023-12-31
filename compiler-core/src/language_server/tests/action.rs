use crate::line_numbers::LineNumbers;
use lsp_types::{
    CodeActionContext, CodeActionParams, PartialResultParams, Position, Range,
    TextDocumentIdentifier, Url, WorkDoneProgressParams, WorkspaceEdit,
};

use super::*;

fn remove_unused_action(src: &str, line: u32) -> String {
    let io = LanguageServerTestIO::new();
    let mut engine = setup_engine(&io);

    // inject stdlib stubs
    _ = io.src_module("list", "");
    _ = io.src_module(
        "result",
        "pub fn is_ok() {}\npub fn is_err() {}\npub fn all() {}",
    );
    _ = io.src_module("map", "pub type Map(key, value)\npub fn delete() {}");
    _ = io.src_module("option", "");

    _ = io.src_module("app", src);
    engine.compile_please().result.expect("compiled");

    // create the code action request
    let path = Utf8PathBuf::from(if cfg!(target_family = "windows") {
        r"\\?\C:\src\app.gleam"
    } else {
        "/src/app.gleam"
    });

    let url = Url::from_file_path(path).unwrap();

    let params = CodeActionParams {
        text_document: TextDocumentIdentifier::new(url.clone()),
        context: CodeActionContext {
            diagnostics: vec![],
            only: None,
            trigger_kind: None,
        },
        range: Range::new(Position::new(0, 0), Position::new(line + 1, 0)),
        work_done_progress_params: WorkDoneProgressParams {
            work_done_token: None,
        },
        partial_result_params: PartialResultParams {
            partial_result_token: None,
        },
    };

    // find the remove unused action response
    let response = engine.action(params).result.unwrap().and_then(|actions| {
        actions
            .into_iter()
            .find(|action| action.title == "Remove unused imports")
    });
    if let Some(action) = response {
        apply_code_action(src, &url, &action)
    } else {
        panic!("No code action produced by the engine")
    }
}

fn suggest_pipeline(src: &str, position_start: Position, position_end: Position) -> String {
    let io = LanguageServerTestIO::new();
    let mut engine = setup_engine(&io);

    _ = io.src_module("list", r#"
            pub fn map(list: List(a), with fun: fn(a) -> b) -> List(b) {
                do_map(list, fun, [])
            }

            fn do_map(list: List(a), fun: fn(a) -> b, acc: List(b)) -> List(b) {
                case list {
                    [] -> reverse(acc)
                    [x, ..xs] -> do_map(xs, fun, [fun(x), ..acc])
                }
            }

            pub fn take_while(
                in list: List(a),
                satisfying predicate: fn(a) -> Bool,
              ) -> List(a) {
                do_take_while(list, predicate, [])
            }

            fn do_take_while(
                list: List(a),
                predicate: fn(a) -> Bool,
                acc: List(a),
              ) -> List(a) {
                case list {
                  [] -> reverse(acc)
                  [first, ..rest] ->
                    case predicate(first) {
                      True -> do_take_while(rest, predicate, [first, ..acc])
                      False -> reverse(acc)
                    }
                }
              }
            
            pub fn reverse(xs: List(a)) -> List(a) {
                do_reverse(xs)
            }
            
            fn do_reverse(list) {
                do_reverse_acc(list, [])
            }
            
            fn do_reverse_acc(remaining, accumulator) {
                case remaining {
                    [] -> accumulator
                    [item, ..rest] -> do_reverse_acc(rest, [item, ..accumulator])
                }
            }

            pub fn is_ok() {}
        "#);
    _ = io.src_module("app", src);
    engine.compile_please().result.expect("compiled");

    // create the code action request
    let path = Utf8PathBuf::from(if cfg!(target_family = "windows") {
        r"\\?\C:\src\app.gleam"
    } else {
        "/src/app.gleam"
    });

    let url = Url::from_file_path(path).unwrap();

    let params = CodeActionParams {
        text_document: TextDocumentIdentifier::new(url.clone()),
        context: CodeActionContext {
            diagnostics: vec![],
            only: None,
            trigger_kind: None,
        },
        range: Range::new(position_start, position_end),
        work_done_progress_params: WorkDoneProgressParams {
            work_done_token: None,
        },
        partial_result_params: PartialResultParams {
            partial_result_token: None,
        },
    };

    let response = engine.action(params).result.unwrap().and_then(|actions| {
        actions
            .into_iter()
            .find(|action| action.title == "Gleam Pipeline suggestion")
    });
    if let Some(action) = response {
        apply_code_action(src, &url, &action)
    } else {
        panic!("No code action produced by the engine")
    }
}

fn apply_code_action(src: &str, url: &Url, action: &lsp_types::CodeAction) -> String {
    match &action.edit {
        Some(WorkspaceEdit { changes, .. }) => match changes {
            Some(changes) => apply_code_edit(src, url, changes),
            None => panic!("No text edit found"),
        },
        _ => panic!("No workspace edit found"),
    }
}

// This function replicates how the text editor applies TextEdit
fn apply_code_edit(
    src: &str,
    url: &Url,
    changes: &HashMap<Url, Vec<lsp_types::TextEdit>>,
) -> String {
    let mut result = src.to_string();
    let line_numbers = LineNumbers::new(src);
    let mut offset = 0;
    dbg!(src);

    for (change_url, change) in changes {
        if url != change_url {
            panic!("Unknown url {}", change_url)
        }
        for edit in change {
            let start =
                line_numbers.byte_index(edit.range.start.line, edit.range.start.character) - offset;
            let end =
                line_numbers.byte_index(edit.range.end.line, edit.range.end.character) - offset;
            let range = (start as usize)..(end as usize);
            offset += end - start;
            result.replace_range(range, &edit.new_text);
        }
    }
    dbg!(result)
}

#[test]
fn test_remove_unused_simple() {
    let code = "
// test
import // comment
  list as lispy
import result
import option

pub fn main() {
  result.is_ok
}
";
    let expected = "
// test

import result


pub fn main() {
  result.is_ok
}
";
    assert_eq!(remove_unused_action(code, 2), expected.to_string())
}

#[test]
fn test_remove_unused_alias() {
    let code = "
// test
import result.{is_ok} as res
import option

pub fn main() {
  is_ok
}
";
    let expected = "
// test
import result.{is_ok}%SPACE%


pub fn main() {
  is_ok
}
";
    assert_eq!(
        remove_unused_action(code, 2),
        expected.replace("%SPACE%", " ")
    )
}

// #[test]
// fn test_suggest_pipeline_assignment1_intermed_var(){

//     // Without Pipeline Operator
//     let code = "
// import list

// fn main() {
//   let x = [1,2,3]
//   let y = list.map(x, fn(x) {x*2}, [])
//   let z = list.take_while(y, fn(x) {x < 3})

//   [1,2,3,4]
// }

// ";

//     // With Pipeline Operator
//     let expected = "
// fn main() {
//   let x = [1, 2, 3]

//   let z =
//     x
//     |> list.map(fn(x) { x * 2 }, [])
//     |> list.take_while(fn(x) { x < 3 })
// }
// ";


//     assert_eq!(suggest_pipeline(code, 1), expected);
// }

#[test]
fn test_suggest_pipeline_assignment_func_chaining(){

    // Without Pipeline Operator
    let code = "
import list

fn main() {
  let result = list.reverse(list.map([1,2,3], fn(x) {x * 2}))
}
";
    // With Pipeline Operator
    let expected = "
import list

fn main() {
  let result = 
[1, 2, 3]
|> list.map(fn(x) { x * 2 })
|> list.reverse()
}
";

    let position_start = Position::new(4, 0);
    let position_end = Position::new(4, 61);


    assert_eq!(suggest_pipeline(code, position_start, position_end), expected);
}

#[test]
fn test_suggest_pipeline_assignment_func_chaining_func_as_input_pipeline(){

    // Without Pipeline Operator
    let code = "
import list

fn main() {
  let result = list.reverse(list.map(buildlist(), fn(x) {x * 2}))
}

fn buildlist() -> List(Int) {
  [1, 2, 3]
}
";
    // With Pipeline Operator
    let expected = "
import list

fn main() {
  let result = 
buildlist()
|> list.map(fn(x) { x * 2 })
|> list.reverse()
}

fn buildlist() -> List(Int) {
  [1, 2, 3]
}
";

    let position_start = Position::new(4, 0);
    let position_end = Position::new(4, 61);


    assert_eq!(suggest_pipeline(code, position_start, position_end), expected);
}

// #[test]
// fn test_suggest_pipeline_on_multiple_lines(){
//     // Without Pipeline Operator
//     let code = "
//     import list

//     fn main() {
//         let init = 1
        
//         let x = [1,2,3]
//         let y = list.map(x, fn(x) {x*2}, [])
//         let z = list.take_while(y, fn(x) {x < 3})
//     }

//     ";

//     // With Pipeline Operator
//     let expected = "
//     fn main() -> Int {
//         let init = 1
        
//         init
//         |> double
//         |> multiplied(4)
//     }
    
//     fn double(in: Int) -> Int {
//         in * 2
//     }
    
//     fn multiplied(mult: Int, in: Int) -> Int {
//         in * mult
//     }
//     ";


//     assert_eq!(suggest_pipeline(code, 1),
//                expected);
//     //Maybe also make this pipeline suggestion on line 2 (doubled)?
//     // assert_eq!(suggest_pipeline(code, 1),
//     // expected)
// }

// #[test]
// fn test_suggest_pipeline_expression(){

//     // Without Pipeline Operator
// //     let code = "
// // import list

// // fn main() -> List(int) {
// //   let x = [1, 2, 3]

// //   x
// //   |> list.map(fn(x) { x * 2 }, [])
// //   |> list.take_while(fn(x) { x < 3 })
// // }
// // ";

// let code = "
// import list

// fn main(){
//     let z = example()
// }

// fn example() -> List(int) {
//     [1, 2, 3, 4, 5]
// }
// ";

//     // With Pipeline Operator
//     let expected = "
// import list

// fn main() -> List(int) {
//   let x = [1, 2, 3]

//   x
//   |> list.map(fn(x) { x * 2 }, [])
//   |> list.take_while(fn(x) { x < 3 })
// }
// ";  
//     assert_eq!(suggest_pipeline(code, 1), expected);
// }

/* TODO: implement qualified unused location
#[test]
fn test_remove_unused_qualified_action() {
    let code = "
// test
import map.{Map, delete}
";
    let expected = "
// test

";
    assert_eq!(remove_unused_action(code), expected.to_string())
}

#[test]
fn test_remove_unused_qualified_partial_action() {
    let code = "
// test
import result.{is_ok, is_err}

pub fn main() {
  is_ok
}
";
    let expected = "
// test
import result.{is_ok}

pub fn main() {
  is_ok
}
";
    assert_eq!(remove_unused_action(code), expected.to_string())
}

#[test]
fn test_remove_unused_qualified_partial2_action() {
    let code = "
// test
import result.{all, is_ok, is_err}

pub fn main() {
  is_ok
}
";
    let expected = "
// test
import result.{ is_ok}

pub fn main() {
  is_ok
}
";
    assert_eq!(remove_unused_action(code), expected.to_string())
}

#[test]
fn test_remove_unused_qualified_partial3_action() {
    let code = "
// test
import result.{all, is_ok, is_err} as res

pub fn main() {
  is_ok
}
";
    let expected = "
// test
import result.{ is_ok} as res

pub fn main() {
  is_ok
}
";
    assert_eq!(remove_unused_action(code), expected.to_string())
}
*/
