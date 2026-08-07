"""Phase 11: Safe calculator — only math, no code execution."""

import ast
import operator
import math as math_module


_ALLOWED_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_ALLOWED_NAMES = {
    "pi": math_module.pi,
    "e": math_module.e,
    "sqrt": math_module.sqrt,
    "sin": math_module.sin,
    "cos": math_module.cos,
    "tan": math_module.tan,
    "log": math_module.log,
    "abs": abs,
    "round": round,
}


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.left), _eval_node(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_OPS:
        return _ALLOWED_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        fn = _ALLOWED_NAMES.get(node.func.id)
        if fn is None:
            raise ValueError(f"Function not allowed: {node.func.id}")
        args = [_eval_node(a) for a in node.args]
        return fn(*args)
    if isinstance(node, ast.Name):
        if node.id in _ALLOWED_NAMES and not callable(_ALLOWED_NAMES[node.id]):
            return _ALLOWED_NAMES[node.id]
        raise ValueError(f"Name not allowed: {node.id}")
    raise ValueError(f"Unsupported expression: {ast.dump(node)}")


def calculate(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        result = _eval_node(tree.body)
        return str(result)
    except Exception as e:
        return f"Error: {e}"
