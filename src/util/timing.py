from __future__ import annotations
from collections import OrderedDict
import time
from timeit import default_timer as timer
from contextlib import contextmanager

import numpy as np
import pandas as pd

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

    def info(self, verbose: bool = False, only_children: bool = False) -> str:
        if verbose:
            message = ""
            if not only_children:
                for i, duration in enumerate(self.durations):
                    message += f"{round(duration * 1000, 2):02} ms ({i + 1}. measurement)\n"
            for k, v in self.children.items():
                message += f"{k}:\n{pad_string(v.info(verbose), "  ")}"
            return message
        else:
            message = f"{round(self.mean_duration * 1000, 2):02} ms ({len(self.durations)} measurements)\n" if len(self.durations) > 0 else "No measurements yet\n" if not only_children else ""
            for k, v in self.children.items():
                message += pad_string(f"{k}: {v.info()}", "  " if not only_children else "")
            return message

def pad_string(to_pad: str, padding: str):
    return (padding + to_pad.replace("\n", "\n" + padding)).rstrip(padding)

class TimingAnalyzer():
    def __init__(self):
        self.root = TimingNode("ROOT")
        self.current_node = self.root
        self.last_time: float | None = None
        self.lines_printed = 0
        pass

    def _start_timing(self):
        self.last_time = timer()

    def _save_time_or_start(self):
        if self.last_time == None:
            self._start_timing()
        else:
            ctime = timer()
            self.current_node.add_duration(ctime - self.last_time)
            self.last_time = ctime

    def sub_category(self, name: str):
        self._save_time_or_start()
        self.current_node = self.current_node.insert_or_get_child(name)

    def next_category(self, name: str):
        self._save_time_or_start()
        try:
            self.current_node = self.current_node.parent.insert_or_get_child(name)
        except:
            raise IndexError("next_category can not be called on root level. Call sub_category first instead.")

    def up(self):
        try:
            self.current_node = self.current_node.parent
        except:
            raise IndexError("up can not be called on root level.")

    def up_next_category(self, name: str):
        self._save_time_or_start()
        try:
            self.current_node = self.current_node.parent.parent.insert_or_get_child(name)
        except:
            raise IndexError("up_next_category can not be called on root level or one level below.")

    @contextmanager
    def context_category(self, name: str):
        self._save_time_or_start()
        cnode = TimingNode(name)
        self.current_node.insert_child(cnode)
        try:
            self.current_node = cnode
        finally:
            self.up()

    def finish_timing(self):
        self._save_time_or_start()
        self.last_time = None
        self.current_node = self.root

    def print(self):
        to_print = self.root.info(only_children=True)
        print(to_print)
        self.lines_printed = to_print.count("\n") + 1

    def print_updating(self):
        for _ in range(self.lines_printed):
            print("\033[A\033[K", end="")
        self.print()

    def _collect_node_results(self, prefix: str, cnode: TimingNode, result_dict: OrderedDict):
        if cnode != self.root: result_dict[prefix + cnode.name] = cnode.durations
        for _, c in cnode.children.items():
            self._collect_node_results(prefix + cnode.name + "->" if cnode != self.root else "", c, result_dict)

    def save_dataframe(self, filename: str):
        results = OrderedDict()
        self._collect_node_results("", self.root, results)
        pd.DataFrame(results).to_csv(filename)
        

if __name__ == "__main__":
    analyzer = TimingAnalyzer()
    for i in range(10):
        analyzer.sub_category("Test 1")
        time.sleep(0.2)
        analyzer.sub_category("Test 1.1")
        time.sleep(0.2)
        analyzer.next_category("Test 1.2")
        analyzer.next_category("Test 1.3")
        time.sleep(0.2)
        analyzer.up_next_category("Test 2")
        time.sleep(0.5)
        analyzer.next_category("Test 3")
        time.sleep(0.1)
        analyzer.finish_timing()
        analyzer.print_updating()
    analyzer.save_dataframe("results/TIMING.csv")