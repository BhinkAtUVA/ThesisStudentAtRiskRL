from __future__ import annotations
from collections import OrderedDict
from collections.abc import Callable
import time
from timeit import default_timer as timer
from contextlib import contextmanager
from typing import Any

import numpy as np
import pandas as pd

type Measurements = tuple[float, float, int] | list[float]

class TimingNode():
    def __init__(self, name: str, children: list[TimingNode] | None = None):
        self.name: str = name
        self.durations: list[float] = []
        self.mean_duration = None
        self.std_duration = None
        self.children: OrderedDict[str, TimingNode] = OrderedDict({ child.name: child for child in children }) if children is not None else OrderedDict()
        self.parent: TimingNode | None = None

    def insert_child(self, child: TimingNode):
        if child.name in self.children: raise ValueError("This name already belongs to a sub category of the current node")
        child.parent = self
        self.children[child.name] = child

    def insert_or_get_child(self, name: str):
        if name in self.children:
            return self.children[name]
        new_node = TimingNode(name)
        new_node.parent = self
        self.children[name] = new_node
        return new_node
    
    def add_duration(self, duration: float):
        self.durations.append(duration)
        self.mean_duration = np.mean(self.durations)
        self.std_duration = np.std(self.durations)

    def _collect_nested(self, all: bool = False) -> tuple[Measurements, list[tuple[str, tuple]]]:
        child_buffer = []
        for k, v in self.children.items():
            child_buffer.append((k, v._collect_nested(all)))
        return (
            self.durations.copy() if all else (self.mean_duration, self.std_duration, len(self.durations)) if len(self.durations) > 0 else RuntimeError("Cannot collect mean measurement if no measurements were made."),
            child_buffer
        )
    
    def _map_nested(buffer: list[tuple[str, tuple[Measurements, list]]], mapper: Callable[[str, Measurements], Any], child_transform: Callable[[Any], Any] = lambda x: x):
        collected = []
        for name, (values, children) in buffer:
            collected.append(mapper(name, values))
            for result in TimingNode._map_nested(children, mapper, child_transform):
                collected.append(child_transform(result))
        return collected

    def _reduce_nested(buffer: list[tuple[str, tuple[Measurements, list]]], reducer: Callable[[Any, float], Any], initial: any = 0):
        result = initial
        for _, (values, children) in buffer:
            result = reducer(result, values[0])
            result = reducer(result, TimingNode._reduce_nested(children, reducer))
        return result
    
    def _collect(self, buffer: list[tuple[str, Measurements]], prefix: str = "", all: bool = True):
        buffer.append((
            prefix + self.name,
            self.durations.copy() if all else (self.mean_duration, self.std_duration, len(self.durations)) if len(self.durations) > 0 else RuntimeError("Cannot collect mean measurement if no measurements were made.")
        ))
        new_prefix = prefix + self.name + "->"
        for _, v in self.children.items():
            v._collect(buffer, new_prefix, all)

    def children_info(self, verbose: bool = False) -> str:
        nested_data = []
        for k, v in self.children.items():
            nested_data.append((k, v._collect_nested(verbose)))

        if verbose:
            def verbose_message(name: str, values: list[float]) -> str:
                message = f"{name}:\n"
                for i, value in enumerate(values):
                    message += f"{round(value * 1000, 2):02}ms ({i + 1}. measurement)\n"
                return message
            message = ""
            for block in TimingNode._map_nested(nested_data, verbose_message, lambda x: pad_string(x, "  ")):
                message += block
            return message
        else:
            total_duration = 0
            for _, (values, _) in nested_data:
                total_duration += values[0]
            def short_message(name: str, values: tuple[float, float, int]) -> str:
                ratio = values[0] / total_duration
                ratio_chars = round(ratio * 20)
                visual = ""
                for _ in range(ratio_chars): visual += "#"
                for _ in range(20 - ratio_chars): visual += "."
                return f"{name}: {round(values[0] * 1000, 2):02}ms (SD {round(values[1] * 1000, 2):02}ms)    {visual} ({round(ratio * 100, 2):02}%), {values[2]} measurements\n"
            message = ""
            for line in TimingNode._map_nested(nested_data, short_message, lambda x: pad_string(x, "  ")):
                message += line
            return message
        
    def children_to_dataframe(self, all_durations: bool = True) -> pd.DataFrame:
        buffer = []
        for _, v in self.children.items():
            v._collect(buffer, all=all_durations)
        transform = OrderedDict()
        max_len = np.max([len(values) for _, values in buffer])
        for name, measurements in buffer:
            transform[name] = measurements
            while(len(transform[name])) < max_len:
                transform[name].append(pd.NA)
        return pd.DataFrame(transform)

def pad_string(to_pad: str, padding: str):
    return (padding + to_pad.replace("\n", "\n" + padding)).rstrip(padding)

class TimingAnalyzer():
    def __init__(self):
        self.root = TimingNode("ROOT")
        self.current_node = self.root
        self.start_times = []
        self.last_time: float | None = None
        self.lines_printed = 0

    def _push_start_time(self):
        self.start_times.append(timer())

    def _pop_push_cmp_time(self):
        ctime = timer()
        diff = ctime - self.start_times.pop()
        self.start_times.append(ctime)
        return diff
    
    def _pop_cmp_time(self):
        return timer() - self.start_times.pop()

    def sub_category(self, name: str):
        self._push_start_time()
        self.current_node = self.current_node.insert_or_get_child(name)

    def next_category(self, name: str):
        self.current_node.add_duration(self._pop_push_cmp_time())
        try:
            self.current_node = self.current_node.parent.insert_or_get_child(name)
        except:
            raise IndexError("next_category can not be called on root level. Call sub_category first instead.")

    def up(self):
        self.current_node.add_duration(self._pop_cmp_time())
        try:
            self.current_node = self.current_node.parent
        except:
            raise IndexError("up can not be called on root level.")

    def up_next_category(self, name: str):
        self.up()
        try:
            self.next_category(name)
        except:
            raise IndexError("up_next_category can not be called on root level or one level below.")

    @contextmanager
    def context_category(self, name: str):
        self._push_start_time()
        cnode = self.current_node.insert_or_get_child(name)
        self.current_node = cnode
        yield
        self.up()

    def finish_timing(self):
        while len(self.start_times) > 0:
            self.up()

    def print(self):
        to_print = self.root.children_info()
        print(to_print)
        self.lines_printed = to_print.count("\n") + 1

    def print_updating(self):
        for _ in range(self.lines_printed):
            print("\033[A\033[K", end="")
        self.print()

    def save_dataframe(self, filename: str, all_durations: bool = True):
        self.root.children_to_dataframe(all_durations).to_csv(filename)
        

if __name__ == "__main__":
    analyzer = TimingAnalyzer()
    for i in range(10):
        analyzer.sub_category("Test 1")
        time.sleep(0.2)
        analyzer.sub_category("Test 1.1")
        time.sleep(0.2)
        with analyzer.context_category("Test 1.2"):
            pass
        analyzer.next_category("Test 1.3")
        time.sleep(0.2)
        analyzer.up_next_category("Test 2")
        time.sleep(0.5)
        analyzer.next_category("Test 3")
        time.sleep(0.1)
        analyzer.finish_timing()
        analyzer.print_updating()
    analyzer.save_dataframe("results/TIMING.csv")