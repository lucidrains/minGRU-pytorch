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

import torch
from torch.utils.data import Dataset, DataLoader

class TokenizedTextDataset(Dataset):
    def __init__(self, data_file, seq_len):
        self.data_file = data_file
        self.seq_len = seq_len
        self.data = np.load(self.data_file, allow_pickle=True)

    def __len__(self):
        return (len(self.data) - self.seq_len) // self.seq_len

    def __getitem__(self, index):
        start_idx = index * self.seq_len
        end_idx = start_idx + self.seq_len + 1
        full_seq = torch.tensor(self.data[start_idx:end_idx], dtype=torch.long)
        return full_seq

    def __getstate__(self):
        state = self.__dict__.copy()
        # Exclude the data attribute from being pickled
        state['data'] = None
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # Re-load the data
        self.data = np.load(self.data_file, allow_pickle=True)


def main(rank, world_size, train_data_file, val_data_file, args):
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

        if rank == 0:
            print(f"Using device: {device}")

        val_data = np.load(val_data_file, allow_pickle=True)
        val_data = val_data.tolist()

        # Constants and Hyperparameters
        NUM_EPOCHS = args.num_epochs  # Number of epochs to train
        BATCH_SIZE = args.batch_size
        GRAD_ACCUM_EVERY = args.grad_accum_every
        LEARNING_RATE = args.learning_rate
        VALIDATE_EVERY = args.validate_every
        CHECKPOINT_EVERY = args.checkpoint_every  # Save checkpoint every n steps
        CHECKPOINT_PATH = args.checkpoint_path  # Path to load checkpoint (if any)
        PRIME_LENGTH = 64
        GENERATE_EVERY = args.generate_every
        GENERATE_LENGTH = 128
        SEQ_LEN = 512

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

        # Create datasets
        train_dataset = TokenizedTextDataset(train_data_file, SEQ_LEN)
        val_dataset = TokenizedTextDataset(val_data_file, SEQ_LEN)

        # Create Distributed Samplers if world_size > 1
        if world_size > 1:
            train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
            val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank)
        else:
            train_sampler = None
            val_sampler = None

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=train_sampler,
                                  shuffle=(train_sampler is None), num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, sampler=val_sampler,
                                shuffle=False, num_workers=0, pin_memory=True)

        # Optimizer
        optim = Adam(model.parameters(), lr=LEARNING_RATE)

        # Mixed Precision Training
        scaler = GradScaler()

        # Initialize training state
        start_epoch = 0
        start_step = 0

        # Checkpoint Loading
        if CHECKPOINT_PATH and os.path.isfile(CHECKPOINT_PATH):
            if rank == 0:
                print(f"Loading checkpoint from {CHECKPOINT_PATH}")
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            start_epoch = checkpoint['epoch']
            start_step = checkpoint['step']
            if world_size > 1:
                # Wrap the model again if needed after loading state_dict
                model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank], output_device=rank)
            if rank == 0:
                print(f"Resumed from epoch {start_epoch}, step {start_step}")

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

        # Training Loop
        for epoch in range(start_epoch, NUM_EPOCHS):
            if world_size > 1:
                train_sampler.set_epoch(epoch)  # For shuffling with DistributedSampler

            if rank == 0:
                print(f"Starting epoch {epoch + 1}/{NUM_EPOCHS}")

            for batch_idx, data in enumerate(train_loader):
                step = start_step + batch_idx
                model.train()
                optim.zero_grad()

                for _ in range(GRAD_ACCUM_EVERY):
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
                    if step % VALIDATE_EVERY == 0:
                        model.eval()
                        with torch.no_grad():
                            data = next(iter(val_loader))
                            data = data.to(device)

                            with autocast():
                                val_loss = model(data, return_loss=True)
                            print(f"Validation loss at epoch {epoch + 1}, step {step}: {val_loss.item():.3f}")
                        model.train()

                    if step % GENERATE_EVERY == 0:
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
                        model.train()

                    # Checkpointing
                    if step % CHECKPOINT_EVERY == 0:
                        checkpoint = {
                            'epoch': epoch,
                            'step': step,
                            'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                            'optimizer_state_dict': optim.state_dict(),
                            'scaler_state_dict': scaler.state_dict(),
                        }
                        checkpoint_filename = f'checkpoint-step-{step}.pt'
                        torch.save(checkpoint, checkpoint_filename)
                        print(f"Checkpoint saved at step {step} to {checkpoint_filename}")

            start_step += len(train_loader)

        # Final Checkpoint after training
        if rank == 0:
            checkpoint = {
                'epoch': NUM_EPOCHS,
                'step': start_step,
                'model_state_dict': model.module.state_dict() if world_size > 1 else model.state_dict(),
                'optimizer_state_dict': optim.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
            }
            checkpoint_filename = f'checkpoint-final.pt'
            torch.save(checkpoint, checkpoint_filename)
            print(f"Final checkpoint saved to {checkpoint_filename}")

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
    import argparse

    parser = argparse.ArgumentParser(description='Train minGRULM with checkpointing and multiple epochs.')

    # Training hyperparameters
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train for.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size per GPU.')
    parser.add_argument('--grad_accum_every', type=int, default=1, help='Gradient accumulation steps.')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate.')
    parser.add_argument('--validate_every', type=int, default=200, help='Validate every n steps.')
    parser.add_argument('--generate_every', type=int, default=300, help='Generate text every n steps.')
    parser.add_argument('--checkpoint_every', type=int, default=1000, help='Save checkpoint every n steps.')
    parser.add_argument('--checkpoint_path', type=str, default='', help='Path to checkpoint to resume training.')
    
    # New arguments for data files
    parser.add_argument('--train_data', type=str, default='train_data.npy', help='Path to training data file.')
    parser.add_argument('--val_data', type=str, default='val_data.npy', help='Path to validation data file.')

    args = parser.parse_args()

    # To run with multiprocessing
    world_size = torch.cuda.device_count()

    if world_size > 1:
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '5554'

        mp.spawn(main,
                 args=(world_size, args.train_data, args.val_data, args),
                 nprocs=world_size,
                 join=True)
    else:
        main(0, world_size, args.train_data, args.val_data, args)
