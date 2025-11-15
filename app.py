"""
Flask web application for Mechanical Design Solver
"""

from flask import Flask, render_template, request, jsonify
import json
import os
from pathlib import Path
import sympy as sp
from sympy import symbols, cos, sin, tan, atan, pi, sqrt
from typing import Dict, List, Any, Optional

class MechanicalModule:
    """Represents a mechanical design module (gears, belts, etc.)"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.variables = {}
        self.equations = []
        self.equation_names = []
        self.known_values = {}
        self.tables = config.get('tables', {})
        
        for var_name, var_info in config['variables'].items():
            self.variables[var_name] = symbols(var_name, real=True, positive=True)
        
        for eq_info in config['equations']:
            eq_str = eq_info['equation']
            eq_name = eq_info['name']
            eq_parsed = sp.sympify(eq_str, locals=self.variables)
            self.equations.append(eq_parsed)
            self.equation_names.append(eq_name)
    
    def set_value(self, var_name: str, value: float):
        if var_name in self.variables:
            self.known_values[var_name] = value
        else:
            raise ValueError(f"Variable {var_name} not found in module")
    
    def get_unknown_vars(self) -> List[str]:
        return [var for var in self.variables.keys() if var not in self.known_values]
    
    def can_solve_equation(self, eq_idx: int) -> Optional[str]:
        eq = self.equations[eq_idx]
        eq_vars = [str(s) for s in eq.free_symbols]
        unknown_in_eq = [v for v in eq_vars if v not in self.known_values]
        if len(unknown_in_eq) == 1:
            return unknown_in_eq[0]
        return None
    
    def solve_one_step(self) -> Optional[Dict[str, Any]]:
        for eq_idx, eq in enumerate(self.equations):
            var_to_solve = self.can_solve_equation(eq_idx)
            
            if var_to_solve:
                eq_substituted = eq
                for known_var, known_val in self.known_values.items():
                    eq_substituted = eq_substituted.subs(self.variables[known_var], known_val)
                
                solutions = sp.solve(eq_substituted, self.variables[var_to_solve])
                
                if solutions:
                    solution = solutions[0]
                    numerical_value = float(solution.evalf())
                    self.known_values[var_to_solve] = numerical_value
                    solved_form = sp.solve(eq, self.variables[var_to_solve])[0]
                    
                    return {
                        'equation_name': self.equation_names[eq_idx],
                        'original_equation': str(eq),
                        'solved_for': var_to_solve,
                        'solved_form': f"{var_to_solve} = {solved_form}",
                        'numerical_result': numerical_value,
                        'substituted_equation': str(solved_form.subs(
                            [(self.variables[k], v) for k, v in self.known_values.items() if k != var_to_solve]
                        ))
                    }
        
        return None
    
    def solve_all(self) -> List[Dict[str, Any]]:
        steps = []
        max_iterations = 100
        iteration = 0
        
        while iteration < max_iterations:
            result = self.solve_one_step()
            if result is None:
                break
            steps.append(result)
            iteration += 1
        
        return steps


def load_module_from_file(filename: str) -> MechanicalModule:
    with open(filename, 'r') as f:
        config = json.load(f)
    return MechanicalModule(config['name'], config)


app = Flask(__name__)

def get_available_modules():
    """Scan for available module JSON files"""
    modules = []
    modules_dir = Path('modules')
    
    # Create modules directory if it doesn't exist
    modules_dir.mkdir(exist_ok=True)
    
    # Look for JSON files in modules directory
    for file_path in modules_dir.glob('*.json'):
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
                modules.append({
                    'id': file_path.stem,
                    'name': config.get('name', file_path.stem),
                    'description': config.get('description', ''),
                    'filename': str(file_path)
                })
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return modules

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/modules', methods=['GET'])
def get_modules():
    """Get list of available modules"""
    modules = get_available_modules()
    return jsonify(modules)

@app.route('/api/module/<module_id>', methods=['GET'])
def get_module_config(module_id):
    """Get configuration for a specific module"""
    modules_dir = Path('modules')
    file_path = modules_dir / f"{module_id}.json"
    
    if not file_path.exists():
        return jsonify({'error': 'Module not found'}), 404
    
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/solve', methods=['POST'])
def solve():
    """Solve equations for given module and inputs"""
    data = request.json
    module_id = data.get('module_id')
    known_values = data.get('known_values', {})
    solve_mode = data.get('mode', 'all')  # 'one' or 'all'
    
    if not module_id:
        return jsonify({'error': 'Module ID required'}), 400
    
    try:
        # Load module
        modules_dir = Path('modules')
        file_path = modules_dir / f"{module_id}.json"
        
        if not file_path.exists():
            return jsonify({'error': 'Module not found'}), 404
        
        module = load_module_from_file(str(file_path))
        
        # Set known values
        for var_name, value in known_values.items():
            if value is not None and value != '':
                try:
                    module.set_value(var_name, float(value))
                except ValueError as e:
                    return jsonify({'error': f'Invalid value for {var_name}: {e}'}), 400
        
        # Solve
        if solve_mode == 'one':
            result = module.solve_one_step()
            if result:
                steps = [result]
            else:
                steps = []
        else:  # solve_mode == 'all'
            steps = module.solve_all()
        
        # Return results
        return jsonify({
            'steps': steps,
            'solved_values': module.known_values,
            'unknown_vars': module.get_unknown_vars()
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset solver state"""
    return jsonify({'success': True})

if __name__ == '__main__':
    # Create modules directory if it doesn't exist
    modules_dir = Path('modules')
    modules_dir.mkdir(exist_ok=True)
    
    print("\n=== Mechanical Design Solver ===")
    print("Server starting on http://127.0.0.1:5000")
    print("\nPlace your module JSON files in the 'modules/' directory")
    print("\nCurrently available modules:")
    available_modules = get_available_modules()
    if available_modules:
        for module in available_modules:
            print(f"  - {module['name']}: {module['description']}")
    else:
        print("  (No modules found - add JSON files to the 'modules/' directory)")
    print("\n")
    
    app.run(debug=True, port=5000)