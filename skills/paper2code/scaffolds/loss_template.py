"""
{{PAPER_TITLE}} — Loss Functions

Paper: https://arxiv.org/abs/{{ARXIV_ID}}
Implements: {{LOSS_DESCRIPTION}}

Section references:
  {{§SECTION}}, Eq. {{N}} — {{DESCRIPTION}}
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class {{LOSS_CLASS}}(nn.Module):
    """§{{SECTION}}, Eq. {{N}} — {{Loss name from paper}}.
    
    "{{Exact quote from paper describing the loss function}}"
    
    Mathematical formulation:
        {{L = formula as written in paper}}
    
    Args:
        {{param}}: {{description}} — §{{SECTION}} specifies {{value}}
    """
    
    def __init__(self, {{params}}):
        super().__init__()
        # §{{SECTION}} — loss hyperparameters
        pass  # REPLACE with actual parameters
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute loss.
        
        Args:
            predictions: model output — shape: (batch, {{dims}})
            targets: ground truth — shape: (batch, {{dims}})
            
        Returns:
            Scalar loss value
        """
        # §{{SECTION}}, Eq. {{N}} — step-by-step following the equation
        # Each intermediate computation gets a shape comment
        
        # NOTE on numerical stability:
        # {{describe any stability considerations}}
        
        pass  # REPLACE with actual loss computation
