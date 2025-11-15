"""
Mechanical Design Equation Solver with Table and Graph Support
Uses SymPy for symbolic equations and scipy for interpolation
"""

import json
import sympy as sp
from sympy import symbols, cos, sin, tan, atan, pi, sqrt
from scipy.interpolate import interp1d, CubicSpline
from scipy.optimize import brentq
from typing import Dict, List, Any, Optional, Union
import numpy as np


class TableLookup:
    """Handles multi-dimensional table lookups with interpolation"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.description = config.get('description', '')
        self.inputs = config['inputs']  # {var_name: {type, interpolation}}
        self.output = config['output']
        self.data = config['data']
        
    def can_solve_forward(self, known_values: Dict[str, Any]) -> bool:
        """Check if all inputs are known and output is unknown"""
        inputs_known = all(inp in known_values for inp in self.inputs.keys())
        output_unknown = self.output not in known_values
        return inputs_known and output_unknown
    
    def can_solve_inverse(self, known_values: Dict[str, Any]) -> Optional[str]:
        """Check if output is known and exactly one input is unknown"""
        if self.output not in known_values:
            return None
        
        unknown_inputs = [inp for inp in self.inputs.keys() if inp not in known_values]
        
        if len(unknown_inputs) == 1:
            return unknown_inputs[0]
        return None
    
    def lookup(self, input_values: Dict[str, Any]) -> float:
        """Perform forward lookup: inputs -> output"""
        # Handle categorical dimensions first
        current_data = self.data
        
        for inp_name, inp_config in self.inputs.items():
            inp_value = input_values[inp_name]
            
            if inp_config['type'] == 'categorical':
                # Exact match for categorical
                if inp_value not in current_data:
                    raise ValueError(f"Categorical value '{inp_value}' not found in table")
                current_data = current_data[inp_value]
        
        # Now handle continuous/integer dimensions
        continuous_inputs = [(name, input_values[name]) 
                            for name, config in self.inputs.items() 
                            if config['type'] != 'categorical']
        
        if len(continuous_inputs) == 0:
            # Pure categorical lookup, data should be a single value
            return float(current_data)
        
        elif len(continuous_inputs) == 1:
            # 1D interpolation
            inp_name, inp_value = continuous_inputs[0]
            x_data = np.array(current_data[inp_name])
            y_data = np.array(current_data[self.output])
            
            interp_method = self.inputs[inp_name].get('interpolation', 'linear')
            
            if interp_method == 'cubic':
                interp_func = CubicSpline(x_data, y_data, extrapolate=False)
            else:
                interp_func = interp1d(x_data, y_data, kind='linear', 
                                      bounds_error=True)
            
            return float(interp_func(inp_value))
        
        else:
            raise NotImplementedError("Multi-dimensional continuous lookup not yet supported")
    
    def inverse_lookup(self, output_value: float, known_inputs: Dict[str, Any], 
                      unknown_input: str) -> float:
        """Perform inverse lookup: output + some inputs -> missing input"""
        
        # For 1D case (output known, find input)
        if len(self.inputs) == 1 and unknown_input in self.inputs:
            # Navigate to correct categorical slice if needed
            current_data = self.data
            
            x_data = np.array(current_data[unknown_input])
            y_data = np.array(current_data[self.output])
            
            # Create interpolation function
            interp_func = interp1d(y_data, x_data, kind='linear', bounds_error=True)
            return float(interp_func(output_value))
        
        # For multi-dimensional case
        # Handle categorical dimensions first
        current_data = self.data
        for inp_name, inp_config in self.inputs.items():
            if inp_config['type'] == 'categorical' and inp_name in known_inputs:
                current_data = current_data[known_inputs[inp_name]]
        
        # Now solve for the unknown continuous input
        x_data = np.array(current_data[unknown_input])
        y_data = np.array(current_data[self.output])
        
        # Create forward interpolation
        forward_func = interp1d(x_data, y_data, kind='linear', bounds_error=True)
        
        # Use root finding to invert
        def objective(x):
            return forward_func(x) - output_value
        
        # Find bounds for search
        x_min, x_max = x_data.min(), x_data.max()
        
        try:
            result = brentq(objective, x_min, x_max)
            return float(result)
        except ValueError:
            raise ValueError(f"Cannot find inverse: output {output_value} outside range")


class GraphLookup:
    """Handles 1D graph curve lookups (simplified TableLookup)"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.description = config.get('description', '')
        self.input = config['input']
        self.output = config['output']
        self.data = np.array(config['data'])  # [[x1, y1], [x2, y2], ...]
        self.bounds = config.get('bounds', {})
        interp_type = config.get('interpolation', 'linear')
        
        # Create interpolation functions
        x_data = self.data[:, 0]
        y_data = self.data[:, 1]
        
        if interp_type == 'cubic':
            self.forward_func = CubicSpline(x_data, y_data, extrapolate=False)
            self.inverse_func = CubicSpline(y_data, x_data, extrapolate=False)
        else:
            self.forward_func = interp1d(x_data, y_data, kind='linear', bounds_error=True)
            self.inverse_func = interp1d(y_data, x_data, kind='linear', bounds_error=True)
    
    def can_solve_forward(self, known_values: Dict[str, Any]) -> bool:
        """Check if input is known and output is unknown"""
        return self.input in known_values and self.output not in known_values
    
    def can_solve_inverse(self, known_values: Dict[str, Any]) -> Optional[str]:
        """Check if output is known and input is unknown"""
        if self.output in known_values and self.input not in known_values:
            return self.input
        return None
    
    def lookup(self, input_value: float) -> float:
        """Forward lookup: input -> output"""
        return float(self.forward_func(input_value))
    
    def inverse_lookup(self, output_value: float) -> float:
        """Inverse lookup: output -> input"""
        return float(self.inverse_func(output_value))


class MechanicalModule:
    """Represents a mechanical design module with equations and data lookups"""
    
    def __init__(self, name: str, config: Dict[str, Any]):
        self.name = name
        self.variables = {}  # {var_name: sympy.Symbol}
        self.variable_info = config['variables']  # Store original variable configs
        self.equations = []  # List of sympy equations
        self.equation_names = []  # Names/descriptions of equations
        self.known_values = {}  # {var_name: numerical_value}
        self.table_lookups = {}  # {name: TableLookup}
        self.graph_lookups = {}  # {name: GraphLookup}
        
        # Create SymPy symbols for all variables
        for var_name, var_info in config['variables'].items():
            self.variables[var_name] = symbols(var_name, real=True, positive=True)
        
        # Parse equations
        for eq_info in config['equations']:
            eq_str = eq_info['equation']
            eq_name = eq_info['name']
            eq_parsed = sp.sympify(eq_str, locals=self.variables)
            self.equations.append(eq_parsed)
            self.equation_names.append(eq_name)
        
        # Load tables and graphs
        tables_config = config.get('tables', {})
        for table_name, table_config in tables_config.items():
            if table_config['type'] == 'table':
                self.table_lookups[table_name] = TableLookup(table_name, table_config)
            elif table_config['type'] == 'graph':
                self.graph_lookups[table_name] = GraphLookup(table_name, table_config)
    
    def set_value(self, var_name: str, value: Union[float, str]):
        """Set a known value for a variable"""
        if var_name in self.variables or var_name in self.variable_info:
            self.known_values[var_name] = value
        else:
            raise ValueError(f"Variable {var_name} not found in module")
    
    def get_unknown_vars(self) -> List[str]:
        """Get list of variables that don't have known values"""
        all_vars = set(self.variables.keys()) | set(self.variable_info.keys())
        return [var for var in all_vars if var not in self.known_values]
    
    def can_solve_equation(self, eq_idx: int) -> Optional[str]:
        """Check if an equation can be solved for exactly one unknown"""
        eq = self.equations[eq_idx]
        eq_vars = [str(s) for s in eq.free_symbols]
        unknown_in_eq = [v for v in eq_vars if v not in self.known_values]
        
        if len(unknown_in_eq) == 1:
            return unknown_in_eq[0]
        return None
    
    def solve_equation(self, eq_idx: int, var_to_solve: str) -> Dict[str, Any]:
        """Solve a specific equation for a variable"""
        eq = self.equations[eq_idx]
        
        # Substitute known values
        eq_substituted = eq
        for known_var, known_val in self.known_values.items():
            if known_var in self.variables:
                eq_substituted = eq_substituted.subs(self.variables[known_var], known_val)
        
        # Solve for unknown
        solutions = sp.solve(eq_substituted, self.variables[var_to_solve])
        
        if solutions:
            solution = solutions[0]
            numerical_value = float(solution.evalf())
            self.known_values[var_to_solve] = numerical_value
            
            solved_form = sp.solve(eq, self.variables[var_to_solve])[0]
            
            return {
                'type': 'equation',
                'equation_name': self.equation_names[eq_idx],
                'original_equation': str(eq),
                'solved_for': var_to_solve,
                'solved_form': f"{var_to_solve} = {solved_form}",
                'numerical_result': numerical_value,
                'substituted_equation': str(solved_form.subs(
                    [(self.variables[k], v) for k, v in self.known_values.items() 
                     if k != var_to_solve and k in self.variables]
                ))
            }
        
        return None
    
    def solve_table(self, table_name: str, direction: str) -> Optional[Dict[str, Any]]:
        """Solve using a table lookup"""
        table = self.table_lookups[table_name]
        
        if direction == 'forward':
            input_values = {inp: self.known_values[inp] for inp in table.inputs.keys()}
            result = table.lookup(input_values)
            self.known_values[table.output] = result
            
            return {
                'type': 'table_lookup',
                'table_name': table_name,
                'description': table.description,
                'direction': 'forward',
                'inputs': input_values,
                'solved_for': table.output,
                'numerical_result': result
            }
        
        elif direction == 'inverse':
            unknown_input = table.can_solve_inverse(self.known_values)
            if not unknown_input:
                return None
            
            known_inputs = {k: v for k, v in self.known_values.items() 
                          if k in table.inputs and k != unknown_input}
            output_value = self.known_values[table.output]
            
            result = table.inverse_lookup(output_value, known_inputs, unknown_input)
            self.known_values[unknown_input] = result
            
            return {
                'type': 'table_lookup',
                'table_name': table_name,
                'description': table.description,
                'direction': 'inverse',
                'output': output_value,
                'known_inputs': known_inputs,
                'solved_for': unknown_input,
                'numerical_result': result
            }
        
        return None
    
    def solve_graph(self, graph_name: str, direction: str) -> Optional[Dict[str, Any]]:
        """Solve using a graph lookup"""
        graph = self.graph_lookups[graph_name]
        
        if direction == 'forward':
            input_value = self.known_values[graph.input]
            result = graph.lookup(input_value)
            self.known_values[graph.output] = result
            
            return {
                'type': 'graph_lookup',
                'graph_name': graph_name,
                'description': graph.description,
                'direction': 'forward',
                'input': {graph.input: input_value},
                'solved_for': graph.output,
                'numerical_result': result
            }
        
        elif direction == 'inverse':
            output_value = self.known_values[graph.output]
            result = graph.inverse_lookup(output_value)
            self.known_values[graph.input] = result
            
            return {
                'type': 'graph_lookup',
                'graph_name': graph_name,
                'description': graph.description,
                'direction': 'inverse',
                'output': output_value,
                'solved_for': graph.input,
                'numerical_result': result
            }
        
        return None
    
    def solve_one_step(self) -> Optional[Dict[str, Any]]:
        """Find and solve one equation, table, or graph"""
        
        # Try equations first
        for eq_idx, eq in enumerate(self.equations):
            var_to_solve = self.can_solve_equation(eq_idx)
            if var_to_solve:
                return self.solve_equation(eq_idx, var_to_solve)
        
        # Try table lookups (forward)
        for table_name, table in self.table_lookups.items():
            if table.can_solve_forward(self.known_values):
                return self.solve_table(table_name, 'forward')
        
        # Try table lookups (inverse)
        for table_name, table in self.table_lookups.items():
            if table.can_solve_inverse(self.known_values):
                return self.solve_table(table_name, 'inverse')
        
        # Try graph lookups (forward)
        for graph_name, graph in self.graph_lookups.items():
            if graph.can_solve_forward(self.known_values):
                return self.solve_graph(graph_name, 'forward')
        
        # Try graph lookups (inverse)
        for graph_name, graph in self.graph_lookups.items():
            if graph.can_solve_inverse(self.known_values):
                return self.solve_graph(graph_name, 'inverse')
        
        return None
    
    def solve_all(self) -> List[Dict[str, Any]]:
        """Iteratively solve all possible equations/tables/graphs"""
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


# Example configuration with Table 9-15 (Lewis Y Factor)
example_with_table = {
    "name": "Spur Gear with Lewis Factor",
    "variables": {
        "N": {"description": "Number of teeth", "unit": "", "type": "integer"},
        "Y": {"description": "Lewis form factor", "unit": "", "type": "continuous"},
        "tooth_form": {"description": "Tooth form type", "unit": "", "type": "categorical",
                      "options": ["20° Full depth", "20° Stub"]},
        "P": {"description": "Diametral pitch", "unit": "1/in", "type": "continuous"},
        "F": {"description": "Face width", "unit": "in", "type": "continuous"},
        "W_t": {"description": "Transmitted load", "unit": "lbf", "type": "continuous"},
        "sigma": {"description": "Bending stress", "unit": "psi", "type": "continuous"}
    },
    "equations": [
        {
            "name": "Lewis bending stress equation",
            "equation": "Eq(sigma, W_t / (F * Y * P))"
        }
    ],
    "tables": {
        "lewis_y_factor": {
            "type": "table",
            "description": "Lewis Tooth Form Factor Y (Table 9-15)",
            "inputs": {
                "N": {"type": "integer", "interpolation": "linear"},
                "tooth_form": {"type": "categorical"}
            },
            "output": "Y",
            "data": {
                "20° Full depth": {
                    "N": [17, 18, 19, 20, 22, 24, 26, 28, 30, 34, 38, 43, 50, 60, 75, 100, 150, 300],
                    "Y": [0.512, 0.521, 0.534, 0.544, 0.559, 0.572, 0.588, 0.597, 0.606, 0.628, 
                          0.651, 0.672, 0.694, 0.713, 0.735, 0.757, 0.779, 0.801]
                },
                "20° Stub": {
                    "N": [14, 15, 16, 17, 18, 19, 20, 22, 24, 26, 28, 30, 34, 38, 43, 50, 60, 75, 100, 150, 300],
                    "Y": [0.540, 0.566, 0.578, 0.587, 0.603, 0.616, 0.628, 0.648, 0.664, 0.678, 
                          0.688, 0.698, 0.714, 0.729, 0.739, 0.758, 0.774, 0.792, 0.808, 0.830, 0.855]
                }
            }
        },
        "contact_stress_grade1": {
            "type": "graph",
            "description": "Allowable contact stress for Grade 1 steel (Fig 9-19)",
            "input": "HB",
            "output": "S_ac",
            "bounds": {"min": 150, "max": 400},
            "interpolation": "linear",
            "data": [
                [150, 600], [200, 700], [250, 775], [300, 850], [350, 925], [400, 1000]
            ]
        }
    }
}


# Example usage
if __name__ == "__main__":
    print("=== Mechanical Solver with Tables & Graphs ===\n")
    
    # Create module
    module = MechanicalModule("Example", example_with_table)
    
    # Example 1: Forward table lookup
    print("Example 1: Finding Y factor from table")
    print("-" * 50)
    module.set_value('N', 25)
    module.set_value('tooth_form', '20° Full depth')
    module.set_value('P', 8)
    module.set_value('F', 2)
    module.set_value('W_t', 500)
    
    steps = module.solve_all()
    for i, step in enumerate(steps, 1):
        if step['type'] == 'table_lookup':
            print(f"Step {i}: Table lookup - {step['description']}")
            print(f"  Direction: {step['direction']}")
            if step['direction'] == 'forward':
                print(f"  Inputs: {step['inputs']}")
            print(f"  Solved: {step['solved_for']} = {step['numerical_result']:.6f}")
        elif step['type'] == 'equation':
            print(f"Step {i}: {step['equation_name']}")
            print(f"  Solved: {step['solved_for']} = {step['numerical_result']:.6f}")
        print()
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Inverse table lookup
    module2 = MechanicalModule("Example2", example_with_table)
    print("Example 2: Finding number of teeth from Y factor (inverse)")
    print("-" * 50)
    module2.set_value('Y', 0.650)
    module2.set_value('tooth_form', '20° Full depth')
    
    steps2 = module2.solve_all()
    for i, step in enumerate(steps2, 1):
        if step['type'] == 'table_lookup':
            print(f"Step {i}: Table lookup - {step['description']}")
            print(f"  Direction: {step['direction']}")
            print(f"  Solved: {step['solved_for']} = {step['numerical_result']:.6f}")
        print()
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Graph lookup
    module3 = MechanicalModule("Example3", example_with_table)
    print("Example 3: Finding allowable stress from hardness (graph)")
    print("-" * 50)
    module3.set_value('HB', 275)
    
    steps3 = module3.solve_all()
    for i, step in enumerate(steps3, 1):
        if step['type'] == 'graph_lookup':
            print(f"Step {i}: Graph lookup - {step['description']}")
            print(f"  Direction: {step['direction']}")
            print(f"  Input: {step['input']}")
            print(f"  Solved: {step['solved_for']} = {step['numerical_result']:.1f}")
        print()