import time
from typing import Dict, List, Optional


class ModelTracker(object):
    def __init__(self, usage: Dict):
        self.usage: Dict = usage

    def __call__(self, stage: str):
        return self.start(stage)

    def start(self, stage: str):
        stage_stats = {
            'stage': stage,
        }
        self.usage['stats'].append(stage_stats)

        start_time = time.perf_counter()

        def done(**kwargs):
            end_time = time.perf_counter()
            duration = int((end_time - start_time) * 1000)
            stage_stats['dur'] = duration
            if 'tokens' in kwargs:
                stage_stats['tokens'] = kwargs['tokens']

        return done


class InferTracker(object):
    def __init__(self):
        self.usages: List[Dict] = []

    def tracker(self, model: str) -> ModelTracker:
        usage = {
            'model': model,
            'stats': [],
        }
        self.usages.append(usage)
        return ModelTracker(usage)
