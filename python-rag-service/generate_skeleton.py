import ast
import os
import glob
import argparse


class SkeletonTransformer(ast.NodeTransformer):
    def transform_function(self, node):
        """Strips function bodies but keeps docstrings intact."""
        docstring = ast.get_docstring(node)
        ellipsis = ast.Expr(value=ast.Constant(value=Ellipsis))

        if docstring:
            doc_node = ast.Expr(value=ast.Constant(value=docstring))
            node.body = [doc_node, ellipsis]
        else:
            node.body = [ellipsis]
        return node

    def visit_FunctionDef(self, node):
        return self.transform_function(node)

    def visit_AsyncFunctionDef(self, node):
        return self.transform_function(node)


def compress_codebase(source_dir: str, output_file: str):
    # Ensure the directory exists before trying to parse it
    if not os.path.exists(source_dir):
        print(f"❌ Error: The directory '{source_dir}' does not exist.")
        return

    transformer = SkeletonTransformer()
    # Normalize paths to handle Windows backslashes cleanly
    search_pattern = os.path.join(source_dir, "**", "*.py")
    python_files = glob.glob(search_pattern, recursive=True)

    if not python_files:
        print(f"⚠️ No Python files found in '{source_dir}'.")
        return

    with open(output_file, "w", encoding="utf-8") as out_f:
        for file_path in python_files:
            # Added exclusions for the generator itself and any generated context files
            if any(exclude in file_path for exclude in ["venv", "__pycache__", "generate_skeleton.py", "context.py"]):
                continue

            with open(file_path, "r", encoding="utf-8") as f:
                source_code = f.read()

            try:
                tree = ast.parse(source_code)
                modified_tree = transformer.visit(tree)
                skeleton_code = ast.unparse(modified_tree)

                out_f.write(f"\n# {'=' * 60}\n")
                out_f.write(f"# FILE: {file_path.replace(os.sep, '/')}\n")  # Clean output paths
                out_f.write(f"# {'=' * 60}\n\n")
                out_f.write(skeleton_code)
                out_f.write("\n")

            except SyntaxError:
                print(f"⚠️ Skipping {file_path} due to syntax error.")

    print(f"✅ Skeleton successfully generated at: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a skeleton context of a Python codebase.")
    parser.add_argument("target", help="The directory you want to map (e.g., app/pipeline/ingestion)")
    parser.add_argument("-o", "--output", default="skeleton_context.py",
                        help="Output file name (default: skeleton_context.py)")

    args = parser.parse_args()
    compress_codebase(args.target, args.output)