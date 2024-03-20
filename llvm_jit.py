import ast

class IRNode:
    def __init__(self, operation, operands):
        self.operation = operation
        self.operands = operands

    def __repr__(self):
        return f"{self.operation}({', '.join(map(str, self.operands))})"

class ASTToIRTransformer(ast.NodeVisitor):
    def __init__(self):
        self.ir = []

    def visit_FunctionDef(self, node):
        for stmt in node.body:
            self.visit(stmt)

    def visit_Return(self, node):
        self.visit(node.value)

    def visit_BinOp(self, node):
        left = self.visit(node.left)
        right = self.visit(node.right)
        op = type(node.op)
        if op == ast.Add:
            operation = 'add'
        elif op == ast.Sub:
            operation = 'sub'
        elif op == ast.Mult:
            operation = 'mul'
        else:
            raise NotImplementedError(f"Operation {op} not implemented in IR")
        temp_var = f"temp{len(self.ir)}"
        self.ir.append(IRNode(operation, (left, right, temp_var)))
        return temp_var

    def visit_Name(self, node):
        return node.id

    def visit_Num(self, node):
        return node.n

    def generate_ir(self, ast_tree):
        self.visit(ast_tree)
        return self.ir

class IROptimizer:
    def __init__(self, ir):
        self.ir = ir

    def optimize(self):
        optimized_ir = []
        seen_operations = {}

        for ir_node in self.ir:
            key = (ir_node.operation, tuple(ir_node.operands[:-1]))
            if key in seen_operations:
                # Replace the operation with the previously computed value
                seen_operations[key].operands[-1] = ir_node.operands[-1]
            else:
                seen_operations[key] = ir_node
                optimized_ir.append(ir_node)

        self.ir = optimized_ir

    def get_optimized_ir(self):
        return self.ir

from llvmlite import ir, binding

class LLVMIRGenerator:
    def __init__(self, optimized_ir):
        self.optimized_ir = optimized_ir
        self.module = ir.Module()
        self.builder = None
        self.function = None
        self.variables = {}

    def generate_function(self):
        # Define the function type
        func_type = ir.FunctionType(ir.IntType(32), [ir.IntType(32), ir.IntType(32)])
        func = ir.Function(self.module, func_type, name="add_multiply")
        block = func.append_basic_block(name="entry")
        self.builder = ir.IRBuilder(block)
        return func

    def process_ir_node(self, ir_node):
        if ir_node.operation == 'add':
            left = self.variables[ir_node.operands[0]]
            right = self.variables[ir_node.operands[1]]
            self.variables[ir_node.operands[2]] = self.builder.add(left, right)
        elif ir_node.operation == 'sub':
            left = self.variables[ir_node.operands[0]]
            right = self.variables[ir_node.operands[1]]
            self.variables[ir_node.operands[2]] = self.builder.sub(left, right)
        elif ir_node.operation == 'mul':
            left = self.variables[ir_node.operands[0]]
            right = self.variables[ir_node.operands[1]]
            self.variables[ir_node.operands[2]] = self.builder.mul(left, right)
        else:
            raise NotImplementedError(f"Operation {ir_node.operation} is not supported in LLVM IR generation")

    def generate_llvm_ir(self):
        self.function = self.generate_function()
        for arg, param in zip(self.optimized_ir[0].operands[:2], self.function.args):
            self.variables[arg] = param

        for ir_node in self.optimized_ir:
            self.process_ir_node(ir_node)

        self.builder.ret(self.variables[self.optimized_ir[-1].operands[2]])
        return str(self.module)


def add_multiply(x, y):
    return (x + y) * (x - y)

tree = ast.parse("def add_multiply(x, y):\n    return (x + y) * (x - y)")
#tree = ast.dump(ast.parse("def add_multiply(x, y):\n    return (x + y) * (x - y)"), indent=4)

print(tree)

# Transform the AST into IR
transformer = ASTToIRTransformer()
intermediate_representation = transformer.generate_ir(tree)

# Print the generated IR
print("Intermediate Representation:")
for ir_node in intermediate_representation:
    print(ir_node)

# Optimize the IR
optimizer = IROptimizer(intermediate_representation)
optimizer.optimize()
optimized_ir = optimizer.get_optimized_ir()

# Print the optimized IR
print("Optimized Intermediate Representation:")
for ir_node in optimized_ir:
    print(ir_node)

# Generate LLVM IR
llvm_ir_generator = LLVMIRGenerator(optimized_ir)
llvm_ir = llvm_ir_generator.generate_llvm_ir()

# Print the generated LLVM IR
print("Generated LLVM IR:")
print(llvm_ir)

# Initialize LLVM
binding.initialize()
binding.initialize_native_target()
binding.initialize_native_asmprinter()

target = binding.Target.from_default_triple()
target_machine = target.create_target_machine()
module = binding.parse_assembly(str(llvm_ir))
module.verify()

# Create a JIT compiler for the module
engine = binding.create_mcjit_compiler(module, target_machine)

import ctypes

# Define the function signature
c_int = ctypes.c_int
CFUNCTYPE = ctypes.CFUNCTYPE

# Execute the function
func_ptr = engine.get_function_address("add_multiply")
cfunc = CFUNCTYPE(c_int, c_int, c_int)(func_ptr)
result = cfunc(5, 3)  # Example arguments
print("Result of add_multiply:", result)

