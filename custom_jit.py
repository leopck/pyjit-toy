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

class MachineCodeGenerator:
    def __init__(self, optimized_ir):
        self.optimized_ir = optimized_ir
        self.assembly_code = []

    def generate_code(self):
        for ir_node in self.optimized_ir:
            if ir_node.operation == 'add':
                self.assembly_code.append(f"ADD {ir_node.operands[0]}, {ir_node.operands[1]}, {ir_node.operands[2]}")
            elif ir_node.operation == 'sub':
                self.assembly_code.append(f"SUB {ir_node.operands[0]}, {ir_node.operands[1]}, {ir_node.operands[2]}")
            elif ir_node.operation == 'mul':
                self.assembly_code.append(f"MUL {ir_node.operands[0]}, {ir_node.operands[1]}, {ir_node.operands[2]}")
            else:
                raise NotImplementedError(f"Operation {ir_node.operation} is not supported in code generation")

    def get_assembly_code(self):
        return self.assembly_code

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

# Generate pseudo-assembly code
code_generator = MachineCodeGenerator(optimized_ir)
code_generator.generate_code()
assembly_code = code_generator.get_assembly_code()

# Print the generated pseudo-assembly code
print("Generated Pseudo-Assembly Code:")
for line in assembly_code:
    print(line)

