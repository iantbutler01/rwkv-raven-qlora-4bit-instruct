from ditty.pipeline import Pipeline
from ditty.data import Data
from datasets import load_dataset
from itertools import chain
from datasets import disable_caching
from consts import PROMPT_WITH_INPUT_FORMAT, PROMPT_NO_INPUT_FORMAT
from collator import DataCollatorForCompletionOnlyLM
import os

class RWKVPipeline(Pipeline):
    def dataset(self):
        data = Data(
            load_kwargs={"path": self.dataset_namespace},
            grad_accum=self.grad_accum,
            batch_size=self.batch_size,
            tokenizer=self.tokenizer,
            group_by_length=True,
            collator=DataCollatorForCompletionOnlyLM(mlm=False, tokenizer=self.tokenizer, return_tensors="pt"),
        )

        def add_text(rec):
            instruction = rec["instruction"]
            response = rec["response"]
            context = rec.get("context")

            if not instruction:
                raise ValueError(f"Expected an instruction in: {rec}")

            if not response:
                raise ValueError(f"Expected a response in: {rec}")

            # For some instructions there is an input that goes along with the instruction, providing context for the
            # instruction.  For example, the input might be a passage from Wikipedia and the instruction says to extract
            # some piece of information from it.  The response is that information to extract.  In other cases there is
            # no input.  For example, the instruction might be open QA such as asking what year some historic figure was
            # born.
            if context:
                rec["text"] = PROMPT_WITH_INPUT_FORMAT.format(
                    instruction=instruction, response=response, input=context
                )
            else:
                rec["text"] = PROMPT_NO_INPUT_FORMAT.format(
                    instruction=instruction, response=response
                )
            return rec

        def tokenize(batch):
            return self.tokenizer(
                batch["text"],
                max_length=self.block_size,
                truncation=True,
            )

        columns = ["instruction", "response", "context", "text", "category"]
        return data.prepare(
            [
                ("map", add_text, {}),
                ("map", tokenize, dict(batched=True, remove_columns=columns)),
                ("filter", lambda rec: len(rec["input_ids"]) < self.block_size, {}),
                ("shuffle", None, dict(seed=self.seed))
            ]
        )


if __name__ == "__main__":
    pipeline = RWKVPipeline(
        dataset_namespace="databricks/databricks-dolly-15k",
        model_name_or_path="mistralai/Mistral-7B-v0.1",
        gradient_checkpointing=False,
        block_size=128,
        grad_accum=8,
        batch_size=1,
        l4bit=False,
        l8bit=False,
        experimental=True,
        fp16=True,
        use_bfloat16=False,
        #use_fsdp=True,
        use_deep_speed=True
        #use_8bit_optim=True
    )

    pipeline.run()
