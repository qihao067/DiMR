"""This file contains the code for speeding up the model training.

Copyright (2024) Bytedance Ltd. and/or its affiliates

Licensed under the Apache License, Version 2.0 (the "License"); 
you may not use this file except in compliance with the License. 
You may obtain a copy of the License at 

    http://www.apache.org/licenses/LICENSE-2.0 

Unless required by applicable law or agreed to in writing, software 
distributed under the License is distributed on an "AS IS" BASIS, 
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. 
See the License for the specific language governing permissions and 
limitations under the License.
"""

from contextlib import contextmanager
from functools import update_wrapper
import os
import threading

import torch


def get_use_compile():
    return os.environ.get("K_DIFFUSION_USE_COMPILE", "1") == "1"


def get_use_flash_attention_2():
    return os.environ.get("K_DIFFUSION_USE_FLASH_2", "1") == "1"


state = threading.local()
state.checkpointing = False


@contextmanager
def checkpointing(enable=True):
    try:
        old_checkpointing, state.checkpointing = state.checkpointing, enable
        yield
    finally:
        state.checkpointing = old_checkpointing


def get_checkpointing():
    return getattr(state, "checkpointing", False)


class compile_wrap:
    def __init__(self, function, *args, **kwargs):
        self.function = function
        self.args = args
        self.kwargs = kwargs
        self._compiled_function = None
        update_wrapper(self, function)

    @property
    def compiled_function(self):
        if self._compiled_function is not None:
            return self._compiled_function
        if get_use_compile():
            try:
                self._compiled_function = torch.compile(self.function, *self.args, **self.kwargs)
            except RuntimeError:
                self._compiled_function = self.function
        else:
            self._compiled_function = self.function
        return self._compiled_function

    def __call__(self, *args, **kwargs):
        return self.compiled_function(*args, **kwargs)