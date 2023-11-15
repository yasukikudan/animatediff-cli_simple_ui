from typing import Callable, Optional

import numpy as np


# Whatever this is, it's utterly cursed.
def ordered_halving(val):
    bin_str = f"{val:064b}"
    bin_flip = bin_str[::-1]
    as_int = int(bin_flip, 2)

    return as_int / (1 << 64)


# I have absolutely no idea how this works and I don't like that.
def uniform(
    step: int = ...,
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    if num_frames <= context_size:
        yield list(range(num_frames))
        return

    context_stride = min(context_stride, int(np.ceil(np.log2(num_frames / context_size))) + 1)

    for context_step in 1 << np.arange(context_stride):
        pad = int(round(num_frames * ordered_halving(step)))
        for j in range(
            int(ordered_halving(step) * context_step) + pad,
            num_frames + pad + (0 if closed_loop else -context_overlap),
            (context_size * context_step - context_overlap),
        ):
            yield [e % num_frames for e in range(j, j + context_size * context_step, context_step)]




def continuous(
    step: int,
    num_steps: Optional[int] = None,
    num_frames: int = 32,
    context_size: Optional[int] = 16,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):

    context_overlap = (context_size//4)+step%3
    context_overlap =context_overlap if context_overlap <context_size//2 else context_size//2

    # num_framesがcontext_size以下の場合、全フレームのインデックスを返す
    if num_frames <= context_size:
        yield list(range(num_frames))
        return
    #1コマ飛び飛びのインデックスを生成
    start_idx = 0  # 開始インデックス

    if num_steps>=10:
        if step<=(num_steps//2):#処理の前半のみに適用
            # stepが偶数の場合、偶数および奇数のインデックスバッチを生成
            if step % 2 == 0:
                # 偶数のインデックスバッチを生成
                start_idx = 0
                while start_idx + context_size * 2 <= num_frames:
                    yield list(range(start_idx, start_idx + context_size * 2, 2))
                    start_idx += context_size * 2
                
                # 偶数のインデックスに対する残り
                if start_idx < num_frames:
                    #num_framesが奇数の場合、1を引いて偶数にする
                    end_idx = num_frames if num_frames%2==0 else num_frames-1
                    end_idx = min(start_idx + context_size * 2, end_idx)
                    yield list(range(end_idx - context_size * 2, end_idx, 2))
            if step % 2 == 1 or step==0:
                # 奇数のインデックスバッチを生成
                # または、stepが0の場合
                start_idx = 1
                while start_idx + context_size * 2 <= num_frames:
                    yield list(range(start_idx, start_idx + context_size * 2, 2))
                    start_idx += context_size * 2
                
                # 奇数のインデックスに対する残り
                if start_idx < num_frames:
                    #num_framesが偶数の場合、1を引いて奇数にする
                    end_idx = num_frames if num_frames%2==1 else num_frames-1
                    end_idx = min(start_idx + context_size * 2, end_idx)
                    yield list(range(end_idx - context_size * 2 + 1, end_idx, 2))


    
    # 連続するフレームのインデックスを生成
    start_idx = 0  # 開始インデックス
    last_sequence_generated = False  # 最後のシーケンスが生成されたかどうかを示すフラグ

    
    while not last_sequence_generated:
        end_idx = start_idx + context_size
        
        # インデックスがnum_framesを超える場合、開始インデックスを調整
        if end_idx > num_frames:
            start_idx = num_frames - context_size
            end_idx = num_frames
            last_sequence_generated = True
        
        yield list(range(start_idx, end_idx))
        start_idx += context_size - context_overlap



def continuous2(
    step: int,
    num_steps: Optional[int] = None,
    num_frames: int = 32,
    context_size: Optional[int] = 16,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    #context_overlap =context_overlap if context_overlap <context_size//2 else context_size//2

    # num_framesがcontext_size以下の場合、全フレームのインデックスを返す
    if num_frames <= context_size:
        yield list(range(num_frames))
        return
    def generate_index_batches(init_idx, context_size, num_frames, interval, skip_remainder=False, overlap=0):
        # 任意の数の倍数で飛び飛びのインデックスバッチを生成
        start_idx = init_idx
        while start_idx + context_size * interval <= num_frames:
            yield list(range(start_idx, start_idx + context_size * interval, interval))
            start_idx += context_size * interval - overlap* interval  # オーバーラップ分だけスタートインデックスを調整

        if skip_remainder:
            return
        
        # インデックスに対する残り
        if start_idx < num_frames:
            # 任意の数の倍数以下に調整
            remainder = num_frames % interval
            end_idx = num_frames - remainder
            end_idx = min(start_idx + context_size * interval, end_idx)
            if end_idx - context_size * interval >= start_idx:
                yield list(range(end_idx - context_size * interval, end_idx, interval))


    if (num_steps-step)>=5:
        if step<=(num_steps//2):#処理の前半のみに適用
            # stepが偶数の場合、偶数および奇数のインデックスバッチを生成
            if step % 2 == 0 or step<=(num_steps//4):
                yield from generate_index_batches(0, context_size, num_frames, 2, overlap=context_size//2)
            if step % 2 == 1 or step<=(num_steps//4):
                yield from generate_index_batches(1, context_size, num_frames, 2, overlap=context_size//2)
            yield from generate_index_batches(step%3,context_size, num_frames,3,overlap=context_size//2)


    
    # 連続するフレームのインデックスを生成
    start_idx = 0  # 開始インデックス
    last_sequence_generated = False  # 最後のシーケンスが生成されたかどうかを示すフラグ

    #動的にオーバーラップを変更する処理
    context_overlap = lambda val: context_size//2
    if step>=(num_steps//2):
        #後半の処理の場合、オーバーラップを変更
        context_overlap = lambda val: context_size//8 +int(ordered_halving(val)*(context_size-context_size//8))

    while not last_sequence_generated:
        end_idx = start_idx + context_size
        
        # インデックスがnum_framesを超える場合、開始インデックスを調整
        if end_idx > num_frames:
            start_idx = num_frames - context_size
            end_idx = num_frames
            last_sequence_generated = True
        
        yield list(range(start_idx, end_idx))
        start_idx += context_size - context_overlap(step*start_idx+step)
        # インデックスがnum_framesと同じ場合、最後のシーケンスが生成されたことを示す
        if end_idx==num_frames:
            last_sequence_generated = True


def continuous3(
    step: int,
    num_steps: Optional[int] = None,
    num_frames: int = 32,
    context_size: Optional[int] = 16,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    #context_overlap =context_overlap if context_overlap <context_size//2 else context_size//2

    # num_framesがcontext_size以下の場合、全フレームのインデックスを返す
    if num_frames <= context_size:
        yield list(range(num_frames))
        return
    def generate_index_batches(init_idx, context_size, num_frames, interval, skip_remainder=False, overlap=0):
        # 任意の数の倍数で飛び飛びのインデックスバッチを生成
        start_idx = init_idx
        while start_idx + context_size * interval <= num_frames:
            yield list(range(start_idx, start_idx + context_size * interval, interval))
            start_idx += context_size * interval - overlap* interval  # オーバーラップ分だけスタートインデックスを調整

        if skip_remainder:
            return
        
        # インデックスに対する残り
        if start_idx < num_frames:
            # 任意の数の倍数以下に調整
            remainder = num_frames % interval
            end_idx = num_frames - remainder
            end_idx = min(start_idx + context_size * interval, end_idx)
            if end_idx - context_size * interval >= start_idx:
                yield list(range(end_idx - context_size * interval, end_idx, interval))


    if (num_steps-step)>=5:
        if step<=(num_steps):#処理の前半のみに適用
            # stepが偶数の場合、偶数および奇数のインデックスバッチを生成
            if step % 2 == 0 or step<=(num_steps):
                yield from generate_index_batches(0, context_size, num_frames, 2, overlap=context_size//2)
            if step % 2 == 1 or step<=(num_steps):
                yield from generate_index_batches(1, context_size, num_frames, 2, overlap=context_size//2)
            yield from generate_index_batches(step%3,context_size, num_frames,3,overlap=context_size//2)


    
    # 連続するフレームのインデックスを生成
    start_idx = 0  # 開始インデックス
    last_sequence_generated = False  # 最後のシーケンスが生成されたかどうかを示すフラグ

    #動的にオーバーラップを変更する処理
    context_overlap = lambda val: context_size//2
    if step>=(num_steps//2):
        #後半の処理の場合、オーバーラップを変更
        context_overlap = lambda val: context_size//8 +int(ordered_halving(val)*(context_size-context_size//8))

    while not last_sequence_generated:
        end_idx = start_idx + context_size
        
        # インデックスがnum_framesを超える場合、開始インデックスを調整
        if end_idx > num_frames:
            start_idx = num_frames - context_size
            end_idx = num_frames
            last_sequence_generated = True
        
        yield list(range(start_idx, end_idx))
        start_idx += context_size - context_overlap(step*start_idx+step)
        # インデックスがnum_framesと同じ場合、最後のシーケンスが生成されたことを示す
        if end_idx==num_frames:
            last_sequence_generated = True



def get_context_scheduler(name: str) -> Callable:
    match name:
        case "uniform":
            return uniform
        case "continuous":
            return continuous
        case "continuous2":
            return continuous2
        case "continuous3":
            return continuous3
        case _:
            raise ValueError(f"Unknown context_overlap policy {name}")


def get_total_steps(
    scheduler,
    timesteps: list[int],
    num_steps: Optional[int] = None,
    num_frames: int = ...,
    context_size: Optional[int] = None,
    context_stride: int = 3,
    context_overlap: int = 4,
    closed_loop: bool = True,
):
    return sum(
        len(
            list(
                scheduler(
                    i,
                    num_steps,
                    num_frames,
                    context_size,
                    context_stride,
                    context_overlap,
                )
            )
        )
        for i in range(len(timesteps))
    )
