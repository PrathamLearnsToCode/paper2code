"""
{{PAPER_TITLE}} — Model Architecture

Paper: https://arxiv.org/abs/{{ARXIV_ID}}
Authors: {{AUTHORS}}
Year: {{YEAR}}

Implements: {{ONE_LINE_DESCRIPTION}}

Section references:
  {{§SECTION_1}} — {{DESCRIPTION_1}}
  {{§SECTION_2}} — {{DESCRIPTION_2}}
  {{§SECTION_3}} — {{DESCRIPTION_3}}

Usage:
    from src.model import {{MODEL_CLASS}}, ModelConfig
    
    config = ModelConfig()
    model = {{MODEL_CLASS}}(config)
    output = model(input_tensor)
"""

import math
from dataclasses import dataclass, field
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    """All model hyperparameters.
    
    Values from {{PAPER_TITLE}} unless marked [UNSPECIFIED].
    Matches configs/base.yaml — change values there, not here.
    """
    # Architecture — §{{ARCH_SECTION}}
    # {{PARAM_1_NAME}}: {{TYPE}} = {{VALUE}}  # §X.Y — "quote from paper"
    # {{PARAM_2_NAME}}: {{TYPE}} = {{VALUE}}  # [UNSPECIFIED] — our choice, alternatives: ...

    pass  # REPLACE with actual config fields


# ---------------------------------------------------------------------------
# Sub-modules
# ---------------------------------------------------------------------------

class {{COMPONENT_A}}(nn.Module):
    """§{{SECTION}} — {{Description of component from paper}}.
    
    "{{Exact quote from paper describing this component}}"
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        # §{{SECTION}} — build layers as described
        pass  # REPLACE with actual layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: {{description}} — shape: (batch, {{dims}})
            
        Returns:
            {{description}} — shape: (batch, {{dims}})
        """
        # §{{SECTION}} — forward pass
        # Every tensor operation gets a shape comment:
        # x = self.linear(x)  # (batch, seq_len, d_model) -> (batch, seq_len, d_ff)
        pass  # REPLACE with actual forward pass


class {{COMPONENT_B}}(nn.Module):
    """§{{SECTION}} — {{Description of component from paper}}."""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        pass  # REPLACE
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass  # REPLACE


# ---------------------------------------------------------------------------
# Main Model
# ---------------------------------------------------------------------------

class {{MODEL_CLASS}}(nn.Module):
    """§{{SECTION}} — {{Paper's name for the full model}}.
    
    Composed of:
      - {{COMPONENT_A}} (§{{SECTION_A}})
      - {{COMPONENT_B}} (§{{SECTION_B}})
    
    "{{Quote from paper describing the overall model}}"
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Build model components
        # REPLACE with actual component instantiation
    
    def forward(
        self,
        x: torch.Tensor,
        # Add other inputs as needed (mask, labels, etc.)
    ) -> torch.Tensor:
        """Forward pass following §{{SECTION}} description.
        
        Args:
            x: {{description}} — shape: (batch, {{input_dims}})
            
        Returns:
            {{description}} — shape: (batch, {{output_dims}})
        """
        # §{{SECTION}} — step-by-step forward pass
        # Mirror the paper's description order
        # Shape comments on every operation
        pass  # REPLACE
    
    def __repr__(self) -> str:
        """Print architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return (
            f"{self.__class__.__name__}(\n"
            f"  config={self.config},\n"
            f"  total_params={total_params:,},\n"
            f"  trainable_params={trainable_params:,}\n"
            f")"
        )
