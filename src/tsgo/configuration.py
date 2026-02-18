import os

import numpy as np

from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from typing import Literal, Optional

from tsgo.container.sample_polyhedron import *

METRICS = Literal['EUCLIDEAN', 'TAXICAB', 'SUP_NORM']

class Config(BaseModel):
    """
    Configuration for TSGO (Time Series Global Optimization).
    """
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    p: int = Field(default=None, description="Number of dispersion points.")
    container: Container = Field(default=None, description="Name of the container to use.")
    D_init: Optional[float] = Field(default=None, description="Initial minimum distance value.")
    packing_density: Optional[float] = Field(default=None, description="Packing density for initial solution generation (if applicable).")
    Q: int = Field(default=3, description="Tabu rearrangement parameter.")
    beta_max: int = Field(default=10, description="Tabu search depth parameter.")
    eps: float = Field(default=1e-8, description="Convergence threshold for optimization.")
    metric: METRICS = Field(default=None, description="Metric to use for distance calculations.")
    iterations: int = Field(default=1, description="Number of iterations for each optimization run.")
    processes: int = Field(default=1, description="Number of parallel processes to use.")
    optimization_method: str = Field(default='L-BFGS-B', description="Gradient-based optimization method to use (e.g., 'L-BFGS-B', 'BFGS').")
    save: bool = Field(default=True, description="Whether to save the optimization results to a file.")
    plot: bool = Field(default=False, description="Whether to generate plots of the optimization results immediately after each run.")
    output_dir: str = Field(default='', description="Directory where optimization results will be saved.")
    verbose: bool = Field(default=False, description="Whether to print detailed logs during optimization.")
    
    @field_validator('p', mode='before')
    @classmethod
    def validate_p(cls, value):
        if value is None or value <= 0:
            raise ValueError("p must be a positive integer.")
        return value
    
    @field_validator('container', mode='before')
    @classmethod
    def validate_container(cls, value):
        if value not in CONTAINERS:
            raise ValueError(f"container must be one of {CONTAINERS}.")
        value = CONTAINERS[value]()
        return value
    
    @field_validator('D_init', mode='before')
    @classmethod
    def validate_D_init(cls, value):
        if value is None or value <= 0:
            raise ValueError("D_init must be a positive float.")
        return value
    
    @model_validator(mode='before')
    @classmethod
    def validate_Q_at_most_p(cls, values):
        Q = values.get('Q')
        p = values.get('p')
        if Q is not None and p is not None and Q > p:
            values['Q'] = p
        return values
    
    @model_validator(mode='before')
    @classmethod
    def convert_packing_density_to_D_init(cls, values):
        packing_density = values.get('packing_density')
        D_init = values.get('D_init')
        if packing_density is None and D_init is None:
            raise ValueError("Either packing_density or D_init must be provided.")
        if packing_density is not None and D_init is not None:
            raise ValueError("Only one of packing_density or D_init should be provided.")
        if packing_density is not None: # convert packing density to D_init
            container_obj = CONTAINERS[values['container']]()
            volume = container_obj.volume
            D_init = ((volume * packing_density) / values['p'] * np.pi * (4/3)) ** (1/3)
            values['D_init'] = D_init
        return values
    
    @model_validator(mode='before')
    @classmethod
    def validate_output_dir(cls, values):
        output_dir = values.get('output_dir')
        if not output_dir:
            values['output_dir'] = os.path.join(
                'output', 
                values['container'],
                f"p{values['p']}",
                values['metric'].lower(),
            )
        os.makedirs(values['output_dir'], exist_ok=True)
        return values
    
    def __repr__(self):
        return (
            f"Config(p={self.p}, container='{self.container}', "
            f"D_init={self.D_init}, Q={self.Q}, beta_max={self.beta_max}, "
            f"eps={self.eps}, metric='{self.metric}', "
            f"iterations={self.iterations}, processes={self.processes}, "
            f"optimization_method='{self.optimization_method}', save={self.save}, "
            f"plot={self.plot}, output_dir='{self.output_dir}', "
            f"verbose={self.verbose})"
        )
        
    def pretty(self, **kwargs):
        return self.model_dump(**kwargs)