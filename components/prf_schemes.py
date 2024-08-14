import torch
from itertools import combinations
from functools import cache

# Key properties of a hashing scheme
HASHING_PROPERTIES = {
    "prf_type": str,  # Name of the PRF mapping tokens to a seed
    "context_width": int,  # Number of previous tokens considered
    "self_salt": bool,  # Use token itself to seed and possibly reject
    "hash_key": int,  # Large prime number for seed generation
}

def lookup_seeding_scheme(seeding_scheme: str):
    if not isinstance(seeding_scheme, str):
        raise ValueError("Seeding scheme should be a string summarizing the procedure.")
    if seeding_scheme in ["simple_1", "lefthash"]:
        prf_type = "additive_prf"
        context_width = 1
        self_salt = False
        hash_key = 15485863
    elif seeding_scheme in ["algorithm-3", "selfhash"]:
        prf_type = "anchored_minhash_prf"
        context_width = 4
        self_salt = True
        hash_key = 15485863
    elif seeding_scheme == "minhash":
        prf_type = "minhash_prf"
        context_width = 4
        self_salt = False
        hash_key = 15485863
    elif seeding_scheme == "skipgram":
        prf_type = "skipgram_prf"
        context_width = 5
        self_salt = False
        hash_key = 15485863
    elif seeding_scheme.startswith("ff"):
        split_scheme = seeding_scheme.split("-")
        prf_type = str(split_scheme[1])
        context_width = int(split_scheme[2])
        self_salt = split_scheme[3] == "True"
        hash_key = int(split_scheme[4]) if len(split_scheme) == 5 else 15485863
    else:
        raise ValueError(f"Invalid seeding scheme name {seeding_scheme} given. Try 'simple_1'?")

    assert prf_type in prf_lookup.keys()
    return prf_type, context_width, self_salt, hash_key


def prf_multiplicative(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.prod().item()


def prf_additive(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.sum().item()


def prf_minfunc(input_ids: torch.LongTensor, salt_key: int) -> int:
    return salt_key * input_ids.min().item()


def prf_simple_skip(input_ids: torch.LongTensor, salt_key: int, k=2) -> int:
    return hash_int(salt_key * input_ids[::k]).prod().item()


def prf_skipgram(input_ids: torch.LongTensor, salt_key: int) -> int:
    return hash_int(salt_key * input_ids[0]).item()


def prf_anchored_skipgram(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    return (hash_int(salt_key * input_ids[0]) * hash_int(salt_key * input_ids[anchor])).item()


def prf_minhash(input_ids: torch.LongTensor, salt_key: int) -> int:
    return hash_int(salt_key * input_ids).min().item()


def prf_anchored_minhash(input_ids: torch.LongTensor, salt_key: int, anchor: int = -1) -> int:
    return (salt_key * hash_int(input_ids) * hash_int(input_ids[anchor])).min().item()


def prf_minskipgram(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    skipgrams = torch.as_tensor(list(combinations(hash_int(salt_key * input_ids), 2)))
    return skipgrams.prod(dim=1).min().item()


def prf_noncommutative(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    key = torch.as_tensor(salt_key, dtype=torch.long)
    for entry in input_ids:
        key *= hash_int(key * entry)
        key %= 2**32
    return key.item()


def prf_positional(input_ids: torch.LongTensor, salt_key: int, k: int = 2) -> int:
    return (salt_key * input_ids * torch.arange(1, len(input_ids) + 1, device=input_ids.device)).sum().item()


prf_lookup = {
    "multiplicative_prf": prf_multiplicative,
    "additive_prf": prf_additive,
    "minfunc_prf": prf_minfunc,
    "simple_skip_prf": prf_simple_skip,
    "skipgram_prf": prf_skipgram,
    "anchored_skipgram_prf": prf_anchored_skipgram,
    "minhash_prf": prf_minhash,
    "anchored_minhash_prf": prf_anchored_minhash,
    "minskipgram_prf": prf_minskipgram,
    "noncomm_prf": prf_noncommutative,
    "position_prf": prf_positional,
}

# Generate a global permutation table once at startup
random_generator = torch.Generator(device=torch.device("cpu"))
random_generator.manual_seed(2971215073)  # Fibonacci 47 is prime
table_size = 1_000_003
global_permutation_table = torch.randperm(table_size, device=torch.device("cpu"), generator=random_generator)


def hash_int(integer_tensor: torch.LongTensor) -> torch.LongTensor:
    return global_permutation_table[integer_tensor.cpu() % table_size] + 1  # Ensure values are on CPU


def avalanche_hash_tensor(integer_tensor: torch.LongTensor):
    i = integer_tensor.to(torch.int32).clone()
    i -= i << 6
    i ^= i >> 17
    i -= i << 9
    i ^= i << 4
    i -= i << 3
    i ^= i << 10
    i ^= i >> 15
    return i.to(torch.long)


@cache
def avalanche_hash_int(integer: int):
    i = integer % (2**32)
    i -= i << 6
    i ^= i >> 17
    i -= i << 9
    i ^= i << 4
    i -= i << 3
    i ^= i << 10
    i ^= i >> 15
    return i
