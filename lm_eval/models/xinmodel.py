from typing import Optional, Union
import os
import sys
import torch
import transformers

import lm_eval.models.utils
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM

# append the path of /lustre/fsw/portfolios/nvr/users/xind/megatron-lm-hymba-distill
sys.path.append("/lustre/fsw/portfolios/nvr/users/xind/megatron-lm-hymba-distill")

from torchtitan.models import model_name_to_cls, model_name_to_tokenizer, models_config

def get_torchtitan_model_sft(
    model_name, 
    model_flavor,
    checkpoint_path=None,
    ):
    model_cls = model_name_to_cls[model_name]
    model_config = models_config[model_name][model_flavor]
    print(f"Building {model_name} {model_flavor} with {model_config}")
    model = model_cls.from_model_args(model_config).get_model()
    # model.init_weights()
    if checkpoint_path is not None:
        state_dict = torch.load(checkpoint_path, map_location="cpu")['model']
        state_dict_new = {}
        for key in state_dict:
            if not key.startswith('layers'):
                new_key = '.'.join(key.split('.')[1:])
                state_dict_new[new_key] = state_dict[key]
        model.load_state_dict(state_dict_new, strict=True)
    return model

@register_model("xinmodel")
class XinLMWrapper(HFLM):
    def __init__(
        self,
        pretrained="/lustre/fsw/portfolios/nvr/users/xind/miscs/models/Qwen2.5-0.5B",
        # To use the HF compatible variant
        is_hf: bool = False,
        **kwargs,
    ) -> None:
        """
        The HFLM arguments

        `backend`, `tokenizer`, `truncation`, `max_length`,
        `device`, `dtype`, `batch_size`, `max_batch_size`, `trust_remote_code`, `use_fast_tokenizer`

        Are all supported by Mamba where they do not conflict
        with Mamba-specific restrictions such as causal LMs only.
        """

        if "backend" in kwargs:
            # mamba currently only supports causal models
            assert kwargs["backend"] == "causal"
        self.is_hf = is_hf 
        super().__init__(
            pretrained=pretrained,
            # set appropriate defaults for tokenizer, max length, etc
            backend=kwargs.pop("backend", "causal"),
            tokenizer=kwargs.pop("tokenizer", "dummy"),
            max_length=kwargs.pop("max_length", 2048),
            **kwargs,
        )


    def _get_config(
        self,
        pretrained: str,
        **kwargs,
    ) -> None:
        if self.is_hf:
            super()._get_config(pretrained, **kwargs)
        else:
            # self._config = load_config_hf(pretrained)
            self._config = None

    def _create_model(
        self,
        pretrained: str,
        dtype: Optional[Union[str, torch.dtype]] = "float16",
        # no `parallelize=True` options
        # no PEFT and quantization options
        # Mamba does not support arbitrary HF from_pretrained() args
        **kwargs,
    ) -> None:
        # input(f"stop here.{kwargs}")
        if self.is_hf:
            super()._create_model(pretrained, dtype=dtype, **kwargs)
        else:
            self._model = get_torchtitan_model_sft(
                model_name=kwargs["model_name"], 
                model_flavor=kwargs["model_flavor"],
                checkpoint_path=kwargs["checkpoint_path"],
            )
            self._model.to(torch.bfloat16).to(self._device)

    def _create_tokenizer(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        tokenizer: Optional[
            Union[
                str,
                transformers.PreTrainedTokenizer,
                transformers.PreTrainedTokenizerFast,
            ]
        ],
        revision: Optional[str] = "main",
        trust_remote_code: Optional[bool] = False,
        use_fast_tokenizer: Optional[bool] = True,
        gguf_file: Optional[str] = None,
        add_bos_token: Optional[bool] = False,
    ) -> None:
        """
        Helper method during initialization.

        Create a tokenizer object corresponding to the correct
        tokenizer for value of `pretrained`, or use the pre-initialized tokenizer passed.
        """
        kwargs = {
            "revision": revision,
            "trust_remote_code": trust_remote_code,
        }

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        if gguf_file is not None:
            kwargs["gguf_file"] = gguf_file
        else:
            kwargs["use_fast"] = use_fast_tokenizer

        if add_bos_token:
            kwargs["add_bos_token"] = True

        if tokenizer:
            if isinstance(tokenizer, str):
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    tokenizer, **kwargs
                )
            else:
                assert isinstance(
                    tokenizer, transformers.PreTrainedTokenizer
                ) or isinstance(tokenizer, transformers.PreTrainedTokenizerFast)
                self.tokenizer = tokenizer
        else:
            # Get tokenizer based on 'pretrained'
            if isinstance(pretrained, str):
                model_name = pretrained
            else:
                # get the HF hub name via accessor on model
                model_name = self.model.name_or_path
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                model_name, **kwargs
            )

        # input(f"tokenizer: {self.tokenizer}")
        return None

    def _model_call(self, inps, attn_mask=None, labels=None):
        # input(
        #     f"\ninps: {inps.shape}"
        #     f"\nattn_mask: {attn_mask}"
        #     f"\nlabels: {labels}"
        # )
        with torch.no_grad():
            if attn_mask is not None or labels is not None:
                assert attn_mask is not None and labels is not None
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForSeq2SeqLM
                return self.model(
                    input_ids=inps, attention_mask=attn_mask, labels=labels
                ).logits
            else:
                assert self.AUTO_MODEL_CLASS == transformers.AutoModelForCausalLM
                return self.model(inps).logits


    def _model_generate(self, context, max_length, stop, **generation_kwargs):

        # print(f"device: {self._device}, context: {context.shape}")

        remove_arg = (
            ["attention_mask"]
        )
        for key in remove_arg:
            if key in generation_kwargs:
                generation_kwargs.pop(key)


        # input(
        #     f"\ncontext: {context.shape}"
        #     f"\nmax_length: {max_length}"
        #     f"\nstop: {stop}"
        #     f"\ngeneration_kwargs: {generation_kwargs}"
        # )


        if not self.is_hf:  # is not HF model
            assert context.size(0) == 1, "only support batch size 1 for now"
            assert generation_kwargs.get("do_sample", False) == False, "only support greedy decoding for now"

            # let's manaully do generation without using cache by directly calling the model 
            for i in range(max_length):
                logits = self._model_call(context)
                next_token = torch.argmax(logits[0, -1, :], dim=-1) # only support greedy decoding now
                # input(f"next_token: {next_token}")
                context = torch.cat([context, next_token.unsqueeze(0).unsqueeze(0)], dim=-1)
                # decode the token to string to check if it is a stop token
                generated_text = self.tokenizer.decode(context[0], skip_special_tokens=False)
                if any(generated_text.endswith(stop_token) for stop_token in stop):
                    break
                
                if context.size(1) >= max_length:
                    break
            

            '''
            output = self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=lm_eval.models.utils.stop_sequences_criteria(
                    self.tokenizer,
                    stop,
                    context.shape[1],
                    context.shape[0],
                ),
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )

            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=False)
            '''

            # input(
            #     f"\ngenerated_text: \n{generated_text}"
            #     f"\noutput: \n{context.shape}"
            # )

            return context
        else:
            stopping_criteria = lm_eval.models.utils.stop_sequences_criteria(
                self.tokenizer,
                stop,
                context.shape[1],
                context.shape[0],
            )
            return self.model.generate(
                input_ids=context,
                max_length=max_length,
                stopping_criteria=stopping_criteria,
                pad_token_id=self.tokenizer.pad_token_id,
                use_cache=True,
                **generation_kwargs,
            )