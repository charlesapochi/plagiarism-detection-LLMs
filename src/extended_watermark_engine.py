from __future__ import annotations
import collections
from math import sqrt
from functools import lru_cache
import scipy.stats
import torch
from tokenizers import Tokenizer
from transformers import LogitsProcessor
from components.normalizers import get_normalizer
from components.prf_schemes import prf_lookup, lookup_seeding_scheme
from components.utils import ngrams


class WatermarkCoreExtended:
    def __init__(
        self,
        vocab: list[int] = None,
        gamma: float = 0.25,
        delta: float = 2.0,
        seeding_scheme: str = "selfhash",
        select_green_tokens: bool = True, 
    ):
        # making sure that seeding_scheme has value
        if seeding_scheme is None:
            seeding_scheme = "selfhash"

        # Vocabulary setup
        self.vocab = vocab
        self.vocab_size = len(vocab)

        # Watermark behavior:
        self.gamma = gamma
        self.delta = delta
        self.rng = None
        self.select_green_tokens = select_green_tokens
        
        self._setup_seed_scheme(seeding_scheme)

    def _setup_seed_scheme(self, seeding_scheme: str) -> None:
        """Initialize seeding strategy settings from the scheme name."""
        self.prf_type, self.context_width, self.self_salt, self.hash_key = lookup_seeding_scheme(seeding_scheme)

    def _initialize_random_generator(self, input_ids: torch.LongTensor) -> None:
        """Seed the RNG using the input context."""
        if input_ids.shape[-1] < self.context_width:
            raise ValueError(f"seeding_scheme requires at least a {self.context_width} token prefix to seed the RNG.")

        prf_key = prf_lookup[self.prf_type](input_ids[-self.context_width:], salt_key=self.hash_key)
        self.rng.manual_seed(prf_key % (2**64 - 1))

    def _generate_greenlist_ids(self, input_ids: torch.LongTensor) -> torch.LongTensor:
        """Generate greenlist ids based on input context."""
        self._initialize_random_generator(input_ids)

        greenlist_size = int(self.vocab_size * self.gamma)
        vocab_permutation = torch.randperm(self.vocab_size, device=input_ids.device, generator=self.rng)
        if self.select_green_tokens:
            greenlist_ids = vocab_permutation[:greenlist_size]
        else:
            greenlist_ids = vocab_permutation[(self.vocab_size - greenlist_size):]
        return greenlist_ids


class LogitsProcessorWithWatermarkExtended(WatermarkCoreExtended, LogitsProcessor):
    def __init__(self, *args, track_spike_entropies: bool = False, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.track_spike_entropies = track_spike_entropies
        self.spike_entropies = None
        if self.track_spike_entropies:
            self._initialize_spike_entropies()

    def _initialize_spike_entropies(self):
        alpha_value = torch.exp(torch.tensor(self.delta)).item()
        gamma_value = self.gamma

        self.z_value = ((1 - gamma_value) * (alpha_value - 1)) / (1 - gamma_value + (alpha_value * gamma_value))
        self.expected_gl_coef = (gamma_value * alpha_value) / (1 - gamma_value + (alpha_value * gamma_value))

        # catch for overflow when bias is "infinite"
        if alpha_value == torch.inf:
            self.z_value = 1.0
            self.expected_gl_coef = 1.0
   
    def _retrieve_spike_entropies(self):
        spike_ents = [[] for _ in range(len(self.spike_entropies))]
        for batch_idx, ent_tensor_list in enumerate(self.spike_entropies):
            for ent_tensor in ent_tensor_list:
                spike_ents[batch_idx].append(ent_tensor.item())
        return spike_ents

    def _retrieve_and_clear_spike_entropies(self):
        spike_ents = self._retrieve_spike_entropies()
        self.spike_entropies = None
        return spike_ents
    
    def _calculate_spike_entropy(self, scores):
        # precomputed z value in init
        probabilities = scores.softmax(dim=-1)
        denoms = 1 + (self.z_value * probabilities)
        renormed_probs = probabilities / denoms
        sum_renormed_probs = renormed_probs.sum()
        return sum_renormed_probs

    def _compute_greenlist_mask(self, scores: torch.FloatTensor, greenlist_token_ids) -> torch.BoolTensor:
        green_tokens_mask = torch.zeros_like(scores, dtype=torch.bool)
        for batch_idx, greenlist in enumerate(greenlist_token_ids):
            if len(greenlist) > 0:
                green_tokens_mask[batch_idx][greenlist] = True
        return green_tokens_mask

    def _apply_bias_to_greenlist_logits(self, scores: torch.Tensor, greenlist_mask: torch.Tensor, greenlist_bias: float) -> torch.Tensor:
        scores[greenlist_mask] = scores[greenlist_mask] + greenlist_bias
        return scores

    def _perform_rejection_sampling(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, tail_rule="fixed_compute") -> list[int]:
        """Generate greenlist based on current candidate next token. Reject and move on if necessary.
        """
        sorted_scores, greedy_predictions = scores.sort(dim=-1, descending=True)

        final_greenlist = []
        for idx, prediction_candidate in enumerate(greedy_predictions):
            greenlist_ids = self._generate_greenlist_ids(torch.cat([input_ids, prediction_candidate[None]], dim=0)) 
            if prediction_candidate in greenlist_ids:  
                final_greenlist.append(prediction_candidate)

            if tail_rule == "fixed_score":
                if sorted_scores[0] - sorted_scores[idx + 1] > self.delta_value:
                    break
            elif tail_rule == "fixed_list_length":
                if len(final_greenlist) == 10:
                    break
            elif tail_rule == "fixed_compute":
                if idx == 40:
                    break
            else:
                pass  
        return torch.as_tensor(final_greenlist, device=input_ids.device)


    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        """Call with previous context as input_ids, and scores for next token."""

        self.rng = torch.Generator(device=input_ids.device) if self.rng is None else self.rng

        list_of_greenlist_ids = [None for _ in input_ids] 
        for batch_idx, input_seq in enumerate(input_ids):
            if self.self_salt:
                greenlist_ids = self._perform_rejection_sampling(input_seq, scores[batch_idx])
            else:
                greenlist_ids = self._generate_greenlist_ids(input_seq)
            list_of_greenlist_ids[batch_idx] = greenlist_ids

            # logic for computing and holding spike entropies for analysis
            if self.track_spike_entropies:
                if self.spike_entropies is None:
                    self.spike_entropies = [[] for _ in range(input_ids.shape[0])]
                self.spike_entropies[batch_idx].append(self._calculate_spike_entropy(scores[batch_idx]))

        green_tokens_mask = self._compute_greenlist_mask(scores=scores, greenlist_token_ids=list_of_greenlist_ids)
        scores = self._apply_bias_to_greenlist_logits(scores=scores, greenlist_mask=green_tokens_mask, greenlist_bias=self.delta)

        return scores    


class WatermarkAnalyzerExtended(WatermarkCoreExtended):
    def __init__(
        self,
        *args,
        device: torch.device = None,
        tokenizer: Tokenizer = None,
        z_threshold: float = 4.0,
        normalizers: list[str] = ["homoglyph"],
        ignore_repeated_ngrams: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        assert device, "Must pass device"
        assert tokenizer, "Need an instance of the generating tokenizer to perform detection"

        self.tokenizer = tokenizer
        self.device = device
        self.z_threshold = z_threshold
        self.rng = torch.Generator(device=self.device)

        self.normalizers = []
        for normalization_strategy in normalizers:
            self.normalizers.append(get_normalizer(normalization_strategy))
        self.ignore_repeated_ngrams = ignore_repeated_ngrams

    def calculate_z_score(self, observed_count, T):
        expected_count = self.gamma
        numer = observed_count - expected_count * T
        denom = sqrt(T * expected_count * (1 - expected_count))
        z = numer / denom
        return z

    def calculate_p_value(self, z):
        p_value = scipy.stats.norm.sf(z)
        return p_value

    @lru_cache(maxsize=2**32)
    def cache_ngram_score(self, prefix: tuple[int], target: int):
        greenlist_ids = self._generate_greenlist_ids(torch.as_tensor(prefix, device=self.device))
        return True if target in greenlist_ids else False

    def score_ngrams(self, input_ids: torch.Tensor):
        if len(input_ids) - self.context_width < 1:
            raise ValueError(
                f"Must have at least {1} token to score after "
                f"the first min_prefix_len={self.context_width} tokens required by the seeding scheme."
            )

        token_ngram_generator = ngrams(input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt)
        frequencies_table = collections.Counter(token_ngram_generator)
        ngram_to_watermark_lookup = {}
        for idx, ngram_example in enumerate(frequencies_table.keys()):
            prefix = ngram_example if self.self_salt else ngram_example[:-1]
            target = ngram_example[-1]
            ngram_to_watermark_lookup[ngram_example] = self.cache_ngram_score(prefix, target)

        return ngram_to_watermark_lookup, frequencies_table

    def generate_green_mask(self, input_ids, ngram_to_watermark_lookup) -> tuple[torch.Tensor]:
        green_token_mask, green_token_mask_unique, offsets = [], [], []
        used_ngrams = {}
        unique_ngram_idx = 0
        ngram_examples = ngrams(input_ids.cpu().tolist(), self.context_width + 1 - self.self_salt)

        for idx, ngram_example in enumerate(ngram_examples):
            green_token_mask.append(ngram_to_watermark_lookup[ngram_example])
            if self.ignore_repeated_ngrams:
                if ngram_example in used_ngrams:
                    pass
                else:
                    used_ngrams[ngram_example] = True
                    unique_ngram_idx += 1
                    green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
            else:
                green_token_mask_unique.append(ngram_to_watermark_lookup[ngram_example])
                unique_ngram_idx += 1
            offsets.append(unique_ngram_idx - 1)
        return (
            torch.tensor(green_token_mask),
            torch.tensor(green_token_mask_unique),
            torch.tensor(offsets),
        )

    def analyze_sequence(
        self,
        input_ids: torch.Tensor,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
    ):
        ngram_to_watermark_lookup, frequencies_table = self.score_ngrams(input_ids)
        green_token_mask, green_unique, offsets = self.generate_green_mask(input_ids, ngram_to_watermark_lookup)

        if self.ignore_repeated_ngrams:
            num_tokens_scored = len(frequencies_table.keys())
            green_token_count = sum(ngram_to_watermark_lookup.values())
        else:
            num_tokens_scored = sum(frequencies_table.values())
            assert num_tokens_scored == len(input_ids) - self.context_width + self.self_salt
            green_token_count = sum(freq * outcome for freq, outcome in zip(frequencies_table.values(), ngram_to_watermark_lookup.values()))
        assert green_token_count == green_unique.sum()

        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=num_tokens_scored))
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=(green_token_count / num_tokens_scored)))
        if return_z_score:
            score_dict.update(dict(z_score=self.calculate_z_score(green_token_count, num_tokens_scored)))
        if return_p_value:
            z_score = score_dict.get("z_score")
            if z_score is None:
                z_score = self.calculate_z_score(green_token_count, num_tokens_scored)
            score_dict.update(dict(p_value=self.calculate_p_value(z_score)))
        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_token_mask.tolist()))
        if return_z_at_T:
            sizes = torch.arange(1, len(green_unique) + 1)
            seq_z_score_enum = torch.cumsum(green_unique, dim=0) - self.gamma * sizes
            seq_z_score_denom = torch.sqrt(sizes * self.gamma * (1 - self.gamma))
            z_score_at_effective_T = seq_z_score_enum / seq_z_score_denom
            z_score_at_T = z_score_at_effective_T[offsets]
            assert torch.isclose(z_score_at_T[-1], torch.tensor(z_score))

            score_dict.update(dict(z_score_at_T=z_score_at_T))

        return score_dict

    def batched_window_scoring(
        self,
        input_ids: torch.Tensor,
        window_size: str,
        window_stride: int = 1,
    ):
        ngram_to_watermark_lookup, frequencies_table = self.score_ngrams(input_ids)
        green_mask, green_ids, offsets = self.generate_green_mask(input_ids, ngram_to_watermark_lookup)
        len_full_context = len(green_ids)

        partial_sum_id_table = torch.cumsum(green_ids, dim=0)

        if window_size == "max":
            sizes = range(1, len_full_context)
        else:
            sizes = [int(x) for x in window_size.split(",") if len(x) > 0]

        z_score_max_per_window = torch.zeros(len(sizes))
        cumulative_eff_z_score = torch.zeros(len_full_context)
        
        s = window_stride

        window_fits = False
        for idx, size in enumerate(sizes):
            if size <= len_full_context:
                window_score = torch.zeros(len_full_context - size + 1, dtype=torch.long)
                window_score[0] = partial_sum_id_table[size - 1]
    
                window_score[1:] = partial_sum_id_table[size::s] - partial_sum_id_table[:-size:s]

                batched_z_score_enum = window_score - self.gamma * size
                z_score_denom = sqrt(size * self.gamma * (1 - self.gamma))
                batched_z_score = batched_z_score_enum / z_score_denom

                maximal_z_score = batched_z_score.max()
                z_score_max_per_window[idx] = maximal_z_score

                z_score_at_effective_T = torch.cummax(batched_z_score, dim=0)[0]
                cumulative_eff_z_score[size::s] = torch.maximum(cumulative_eff_z_score[size::s], z_score_at_effective_T[:-1])
                window_fits = True  

        if not window_fits:
            raise ValueError(
                f"Could not find a fitting window with window sizes {window_size} for (effective) context length {len_full_context}."
            )

        cumulative_z_score = cumulative_eff_z_score[offsets]
        optimal_z, optimal_window_size_idx = z_score_max_per_window.max(dim=0)
        optimal_window_size = sizes[optimal_window_size_idx]
        return (
            optimal_z,
            optimal_window_size,
            z_score_max_per_window,
            cumulative_z_score,
            green_mask,
        )

    def analyze_sequence_with_windows(
        self,
        input_ids: torch.Tensor,
        window_size: str = None,
        window_stride: int = 1,
        return_num_tokens_scored: bool = True,
        return_num_green_tokens: bool = True,
        return_green_fraction: bool = True,
        return_green_token_mask: bool = False,
        return_z_score: bool = True,
        return_z_at_T: bool = True,
        return_p_value: bool = True,
    ):
        (
            optimal_z,
            optimal_window_size,
            _,
            z_score_at_T,
            green_mask,
        ) = self.batched_window_scoring(input_ids, window_size, window_stride)
        
        score_dict = dict()
        if return_num_tokens_scored:
            score_dict.update(dict(num_tokens_scored=optimal_window_size))

        denom = sqrt(optimal_window_size * self.gamma * (1 - self.gamma))
        green_token_count = int(optimal_z * denom + self.gamma * optimal_window_size)
        green_fraction = green_token_count / optimal_window_size
        if return_num_green_tokens:
            score_dict.update(dict(num_green_tokens=green_token_count))
        if return_green_fraction:
            score_dict.update(dict(green_fraction=green_fraction))
        if return_z_score:
            score_dict.update(dict(z_score=optimal_z))
        if return_z_at_T:
            score_dict.update(dict(z_score_at_T=z_score_at_T))
        if return_p_value:
            z_score = score_dict.get("z_score", optimal_z)
            score_dict.update(dict(p_value=self._compute_p_value(z_score)))

        if return_green_token_mask:
            score_dict.update(dict(green_token_mask=green_mask.tolist()))

        return score_dict
    
    
    def analyze(
        self,
        text: str = None,
        tokenized_text: list[int] = None,
        window_size: str = None,
        window_stride: int = None,
        return_prediction: bool = True,
        return_scores: bool = True,
        z_threshold: float = None,
        convert_to_float: bool = False,
        **kwargs,
    ) -> dict:
        
        assert (text is not None) ^ (tokenized_text is not None), "Must pass either the raw or tokenized string"
        if return_prediction:
            kwargs["return_p_value"] = True 

        for normalizer in self.normalizers:
            text = normalizer(text)
        if len(self.normalizers) > 0:
            print(f"Text after normalization:\n\n{text}\n")

        if tokenized_text is None:
            assert self.tokenizer is not None, (
                "Watermark detection on raw string ",
                "requires an instance of the tokenizer ",
                "that was used at generation time.",
            )
            tokenized_text = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)["input_ids"][0].to(self.device)
            if tokenized_text[0] == self.tokenizer.bos_token_id:
                tokenized_text = tokenized_text[1:]
        else:
            if (self.tokenizer is not None) and (tokenized_text[0] == self.tokenizer.bos_token_id):
                tokenized_text = tokenized_text[1:]


        output_dict = {}
        if window_size is not None:
            score_dict = self.analyze_sequence_with_windows(
                tokenized_text,
                window_size=window_size,
                window_stride=window_stride,
                **kwargs,
            )
            output_dict.update(score_dict)
        else:
            score_dict = self.analyze_sequence(tokenized_text, **kwargs)
        if return_scores:
            output_dict.update(score_dict)
        
        if return_prediction:
            z_threshold = z_threshold if z_threshold else self.z_threshold
            assert z_threshold is not None, "Need a threshold in order to decide outcome of detection test"
            output_dict["prediction"] = score_dict["z_score"] > z_threshold
            if output_dict["prediction"]:
                output_dict["confidence"] = 1 - score_dict["p_value"]

        if convert_to_float:
            for key, value in output_dict.items():
                if isinstance(value, int):
                    output_dict[key] = float(value)

        return output_dict