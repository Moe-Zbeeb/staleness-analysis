from prime_rl.orchestrator.train_sink import TrainSink


class ProvenanceTrainSink(TrainSink):
    def __init__(self, original):
        if original.pending_batch or original.buffered_count():
            raise ValueError("Install sample provenance before collecting a cohort")
        super().__init__(
            original.config,
            tokenizer=original.tokenizer,
            train_envs=original.train_envs,
            progress=original.progress,
            batch_size=original.batch_size,
            token_batch_size=original.token_batch_size,
            on_result=original.on_result,
        )
        self.sample_provenance = ()

    def process_batch(self):
        owners = {
            id(sample): self.episode_by_trace[trace_id]
            for trace_id, samples in self.pending_batch.items()
            for sample in samples
        }
        batch = super().process_batch()
        self.sample_provenance = tuple(
            (str(owners[id(sample)].id), str(owners[id(sample)].task.data.question_id)) for sample in batch.samples
        )
        if len({response for response, _ in self.sample_provenance}) != len(batch.samples):
            raise ValueError("The study requires exactly one training sample per response")
        return batch
