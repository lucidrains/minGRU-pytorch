import math
import random
import tqdm
import numpy as np
from torch import Tensor
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
import os
import sentencepiece as spm

from minGRU_pytorch.minGRULM import minGRULM

def main(rank, world_size, train_data, val_data):
    try:
        # Initialize the process group only if world_size > 1
        if world_size > 1:
            dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=world_size)
            # Set the device for each process
            torch.cuda.set_device(rank)
            device = torch.device(f'cuda:{rank}')
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            rank = 0  # Since there's only one process

        print(f"Using device: {device}")

        # Constants
        NUM_BATCHES = int(1e5)
        BATCH_SIZE = 8
        GRAD_ACCUM_EVERY = 1
        LEARNING_RATE = 1e-4
        VALIDATE_EVERY = 100
        PRIME_LENGTH = 128
        GENERATE_EVERY = 500
        GENERATE_LENGTH = 256
        SEQ_LEN = 1024

        # Load the SentencePiece tokenizer
        sp = spm.SentencePieceProcessor()
        sp.load('tokenizer.model')

        # Get vocabulary size
        vocab_size = sp.get_piece_size()
        if rank == 0:
            print(f"Vocabulary size: {vocab_size}")

        # The minGRU language model
        model = minGRULM(
            num_tokens=vocab_size,
            dim=1024,
            depth=12
        ).to(device)

        # Wrap the model with DDP only if world_size > 1
        if world_size > 1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)

        # Define the dataset class
        class TokenizedTextDataset(Dataset):
            def __init__(self, data, seq_len):
                self.data = data
                self.seq_len = seq_len

            def __len__(self):
                return (len(self.data) - self.seq_len) // self.seq_len

            def __getitem__(self, index):
                start_idx = index * self.seq_len
                end_idx = start_idx + self.seq_len + 1
                full_seq = torch.tensor(self.data[start_idx:end_idx], dtype=torch.long)
                return full_seq

        # Create datasets
        train_dataset = TokenizedTextDataset(train_data, SEQ_LEN)
        val_dataset = TokenizedTextDataset(val_data, SEQ_LEN)

        # Create Distributed Samplers if world_size > 1
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        else:
            train_sampler = None
            val_sampler = None

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=2, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler, shuffle=False, num_workers=2, pin_memory=True)

        # Optimizer
        optim = Adam(model.parameters(), lr=LEARNING_RATE)

        # Mixed Precision Training
        scaler = GradScaler()

        # Sampling helpers (same as before)
        def log(t, eps=1e-20):
            return torch.log(t.clamp(min=eps))

        def gumbel_noise(t):
            noise = torch.zeros_like(t).uniform_(0, 1)
            return -log(-log(noise))

        def gumbel_sample(t, temperature=1.0, dim=-1):
            return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)

        def top_k(logits, thres=0.9):
            k = max(1, int((1 - thres) * logits.shape[-1]))
            val, ind = torch.topk(logits, k)
            probs = torch.full_like(logits, float('-inf'))
            probs.scatter_(-1, ind, val)
            return probs

        def base_decoding(
            net,
            prompt: Tensor,
            seq_len: int,
            temperature=1.0,
            filter_thres=0.9,
        ):
            prompt_seq_len, out = prompt.shape[-1], prompt.clone()
            sample_num_times = max(0, seq_len - prompt_seq_len)

            for _ in range(sample_num_times):
                logits = net(out)
                logits = logits[:, -1]

                logits = top_k(logits, thres=filter_thres)
                sample = gumbel_sample(logits, temperature=temperature, dim=-1)

                out = torch.cat((out, sample[..., None]), dim=-1)

            return out[..., :]

        def decode_tokens(token_ids):
            return sp.decode(token_ids)

        # Training
        for i in range(NUM_BATCHES):
            model.train()
            if world_size > 1 and train_sampler is not None:
                train_sampler.set_epoch(i)  # Set epoch for reproducibility with DistributedSampler
            optim.zero_grad()

            for _ in range(GRAD_ACCUM_EVERY):
                data = next(iter(train_loader))
                data = data.to(device)

                with autocast():
                    loss = model(data, return_loss=True)
                    loss = loss / GRAD_ACCUM_EVERY

                scaler.scale(loss).backward()

            # Gradient clipping and optimizer step
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            scaler.step(optim)
            scaler.update()
            optim.zero_grad()

            # Validation and Generation (only on rank 0)
            if rank == 0:
                if i % VALIDATE_EVERY == 0:
                    model.eval()
                    with torch.no_grad():
                        data = next(iter(val_loader))
                        data = data.to(device)

                        with autocast():
                            val_loss = model(data, return_loss=True)
                        print(f"Validation loss at step {i}: {val_loss.item():.3f}")

                if i % GENERATE_EVERY == 0:
                    model.eval()
                    with torch.no_grad():
                        # Sample a random starting point in validation data
                        start_idx = random.randint(0, len(val_data) - PRIME_LENGTH - 1)
                        inp = torch.tensor(val_data[start_idx:start_idx + PRIME_LENGTH], dtype=torch.long).to(device)

                        prime = decode_tokens(inp.tolist())
                        print(f"Prime text:\n{prime}\n{'*' * 100}")

                        prompt = inp.unsqueeze(0)  # Add batch dimension

                        sampled = base_decoding(model, prompt, PRIME_LENGTH + GENERATE_LENGTH)
                        sampled_ids = sampled[0].tolist()[PRIME_LENGTH:]  # Exclude the prime tokens

                        base_decode_output = decode_tokens(sampled_ids)

                        print(f"Generated text:\n{base_decode_output}\n")

        # Clean up
        if world_size > 1:
            dist.destroy_process_group()

    except Exception as e:
        print(f"Exception in process {rank}: {e}")
        import traceback
        traceback.print_exc()
        # Re-raise the exception if you want the process to terminate
        raise e

if __name__ == '__main__':
    # Data preparation and tokenizer training

    from datasets import load_dataset

    # Load the 'minipile' dataset
    print("Loading the dataset...")
    dataset = load_dataset('JeanKaddour/minipile', split='train')

    import os
    from tqdm import tqdm

    # Ensure the output directory exists
    os.makedirs('data', exist_ok=True)

    # Save the dataset to a text file
    print("Saving the dataset to 'data/dataset.txt'...")
    with open('data/dataset.txt', 'w', encoding='utf-8') as f:
        for example in tqdm(dataset, desc="Saving texts"):
            text = example['text'].strip()
            if text:
                f.write(text + '\n')

    import sentencepiece as spm

    print("Training the SentencePiece tokenizer on 'data/dataset.txt'...")
    spm.SentencePieceTrainer.Train(
        input='data/dataset.txt',
        model_prefix='tokenizer',
        vocab_size=32768,  # Adjust vocab size as needed
        character_coverage=1.0,
        model_type='bpe'  # You can also try 'unigram' or other types
    )

    print("Loading the trained SentencePiece tokenizer...")
    sp = spm.SentencePieceProcessor()
    sp.load('tokenizer.model')

    # Get vocabulary size
    vocab_size = sp.get_piece_size()
    print(f"Vocabulary size: {vocab_size}")

    from multiprocessing import Pool
    from itertools import chain

    def tokenize_batch(batch_texts):
        return sp.encode(batch_texts, out_type=int)

    # Extract texts from the dataset
    texts = [example['text'].strip() for example in dataset if example['text'].strip()]

    batch_size = 1000  # Adjust based on your memory constraints
    num_processes = 16  # Adjust based on your CPU cores

    # Split texts into batches
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]

    print("Tokenizing the dataset using batch processing and multiprocessing...")
    with Pool(processes=num_processes) as pool:
        tokenized_batches = list(tqdm(
            pool.imap(tokenize_batch, batches),
            total=len(batches),
            desc="Tokenizing"
        ))

    # Flatten the tokenized data
    print("Flattening token IDs...")
    tokenized_data = list(chain.from_iterable(chain.from_iterable(tokenized_batches)))

    # Split the tokenized data into training and validation sets
    split_idx = int(len(tokenized_data) * 0.9)
    train_data = tokenized_data[:split_idx]
    val_data = tokenized_data[split_idx:]

    # Uncomment the following lines to test without multiprocessing
    '''
    world_size = 1
    main(0, world_size, train_data, val_data)
    '''

    # To run with multiprocessing
    world_size = torch.cuda.device_count()

    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '12355'

        mp.spawn(main,
                 args=(world_size, train_data, val_data),
                 nprocs=world_size,
                 join=True)
    else:
        main(0, world_size, train_data, val_data)
