# Copyright (c) BossNET. and affiliates.
# This software may be used and distributed according to the terms of the BoSS Community License Agreement.

from dataclasses import dataclass, field
from typing import ClassVar, List

@dataclass
class lora_config:
     r: int=8
     lora_alpha: int=32
     target_modules: ClassVar[List[str]]= ["q_proj", "v_proj"]
     bias= "none"
     task_type: str= "CAUSAL_LM"
     lora_dropout: float=0.05
     inference_mode: bool = False

@dataclass
class boss_adapter_config:
     adapter_len: int= 10
     adapter_layers: int= 30
     task_type: str= "CAUSAL_LM"

@dataclass
class prefix_config:
     num_virtual_tokens: int=30
     task_type: str= "CAUSAL_LM"    