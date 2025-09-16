# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time
import torch
from contextlib import contextmanager

class Timer:
    def __init__(self, name):
        self.name = name
        self.total_time = 0
        self.start_time = None
    
    def tic(self):
        self.start_time = time.time()
    
    def toc(self):
        if self.start_time is not None:
            self.total_time += time.time() - self.start_time
            self.start_time = None
    
    def reset(self):
        self.start_time = None
        self.total_time = 0
    
    def print(self, string_mode = False, in_second = True):
        if in_second:
            timing_report = "time({}): {:.3f} sec".format(self.name, self.total_time)
        else:
            timing_report = "time({}): {:.3f} min".format(self.name, self.total_time / 60)
        
        if string_mode:
            return timing_report
        else:
            print(timing_report)
    
class TimeReport:
    def __init__(self, cuda_synchronize = False):
        self.timers = {}
        self.cuda_synchronize = cuda_synchronize
    
    def add_timer(self, timer_name):
        self.timers[timer_name] = Timer(timer_name)
    
    def add_timers(self, timer_names):
        for timer_name in timer_names:
            self.add_timer(timer_name)
            
    def reset_timer(self, timer_name = None):
        if timer_name is None:
            for timer_name in self.timers:
                self.timers[timer_name].reset()
        else:
            assert timer_name in self.timers
            self.timers[timer_name].reset()
    
    def delete_timer(self, timer_name):
        assert timer_name in self.timers
        del self.timers[timer_name]
    
    def start_timer(self, timer_name):
        assert timer_name in self.timers
        if self.cuda_synchronize:
            torch.cuda.synchronize()
        self.timers[timer_name].tic()
    
    def end_timer(self, timer_name):
        assert timer_name in self.timers
        if self.cuda_synchronize:
            torch.cuda.synchronize()
        self.timers[timer_name].toc()
    
    def print(self, string_mode = False, in_second = True):
        timing_summary = ""
        for timer_idx, (_, timer) in enumerate(self.timers.items()):
            timing_report = timer.print(string_mode = True, in_second = in_second)
            if timer_idx == 0:
                timing_summary = timing_report
            else:
                timing_summary = timing_summary + ", " + timing_report
        
        if string_mode:
            return timing_summary
        else:
            print(timing_summary)

@contextmanager
def TimeProfiler(time_report: TimeReport, timer_name: str):
    time_report.start_timer(timer_name)
    yield
    time_report.end_timer(timer_name)