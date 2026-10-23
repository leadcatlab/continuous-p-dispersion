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
    Q: Optional[int] = Field(default=None, description="Tabu neighbourhood size; None = max(3, round(ln p)).")
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
    seed: Optional[int] = Field(default=None, description="Master RNG seed. Every stochastic step derives from it, so a run is reproducible from (seed, p, container, metric).")
    
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
            # p balls of radius D fill volume * density:
            #   p * (4/3) * pi * D^3 = volume * density
            # The previous form multiplied by (4/3)*pi instead of dividing,
            # overshooting D_init by (4*pi/3)^(2/3) ~ 2.6x. An initial radius
            # above the true optimum leaves Tabu Search chasing an infeasible
            # target for beta_max iterations before AdjustDistance can back off.
            import tsgo.algorithm.energy_functions as _ef
            D_init = _ef.initial_radius(container_obj, values['p'], packing_density,
                                        values.get('metric') or 'EUCLIDEAN')
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
            f"verbose={self.verbose}, seed={self.seed})"
        )
        
    def pretty(self, **kwargs):
        return self.model_dump(**kwargs)