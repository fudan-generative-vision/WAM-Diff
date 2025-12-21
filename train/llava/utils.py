"""
Various utility functions.
"""
import io
import logging
import logging.handlers
import os
import random
import sys

import numpy as np
import requests
from PIL import Image

import llava.s3_ops as st
from llava.constants import LOGDIR

import cv2
import torch.distributed as dist

try:
    import av
    
except ImportError:
    print("Please install pyav to use video processing functions.")
try:
    from decord import VideoReader, cpu
except ImportError:
    print("Be care decord is not install")

import imageio.v2 as imageio

handler = None


def process_gif(gif_file):
    """
    Simple GIF loader using imageio and open_file.
    Returns:
        video: np.ndarray (num_frames, H, W, C)
        video_time: total duration in seconds
        frame_time: comma-separated frame timestamps ("0.00s,0.10s,...")
        num_frames_to_sample: number of frames
    """
    # open file (supports remote or local)
    f = st.open_file(gif_file, "rb")
    data = f.read()
    f.close()

    bio = io.BytesIO(data)
    reader = imageio.get_reader(bio, format="GIF")

    # Get metadata to estimate total duration
    meta = reader.get_meta_data()
    total_duration = meta.get('duration', 0) / 1000.0  # ms → s

    # --- Fetch only the first frame ---
    first_frame = reader.get_data(0)
    frame_array = np.expand_dims(first_frame, axis=0)  # shape: (1, H, W, C)
    # ----------------------------------

    frame_time = "0.00s"
    num_frames_to_sample = 1

    reader.close()

    return frame_array, total_duration, frame_time, num_frames_to_sample


def process_video_with_decord(video_file):
    
    video_file = st.open_file(video_file, mode = "rb")

    vr = VideoReader(video_file, ctx=cpu(0), num_threads=1)

    if len(vr) == 0:
        raise ValueError("Empty video file")
    
    first_frame = vr.get_batch([0]).asnumpy()

    fps = vr.get_avg_fps() or 0.0

    video_time = (len(vr) / fps) if fps else 0.0
    frame_time = "0.00s"

    num_frames_to_sample = 1

    vr.seek(0)

    return first_frame, video_time, frame_time, num_frames_to_sample

def process_video_with_pyav(video_file, data_args):
    container = av.open(video_file)
    # !!! This is the only difference. Using auto threading
    container.streams.video[0].thread_type = "AUTO"

    video_frames = []
    for packet in container.demux():
        if packet.stream.type == 'video':
            for frame in packet.decode():
                video_frames.append(frame)
    total_frame_num = len(video_frames)
    video_time = video_frames[-1].time
    avg_fps = round(total_frame_num / video_time / data_args.video_fps)
    frame_idx = [i for i in range(0, total_frame_num, avg_fps)]

    if data_args.frames_upbound > 0:
        if len(frame_idx) > data_args.frames_upbound:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, data_args.frames_upbound, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()


    frames = [video_frames[i] for i in frame_idx]
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])

def get_frame_indices(num_frames, vlen, sample='rand', fix_start=None, input_fps=1, max_num_frames=-1):
    if sample in ['rand', 'middle']: # uniform sampling
        acc_samples = min(num_frames, vlen)
        # split the video into `acc_samples` intervals, and sample from each interval.
        intervals = np.linspace(start=0, stop=vlen, num=acc_samples + 1).astype(int)
        ranges = []
        for idx, interv in enumerate(intervals[:-1]):
            ranges.append((interv, intervals[idx + 1] - 1))
        if sample == 'rand':
            try:
                frame_indices = [random.choice(range(x[0], x[1])) for x in ranges]
            except:
                frame_indices = np.random.permutation(vlen)[:acc_samples]
                frame_indices.sort()
                frame_indices = list(frame_indices)
        elif fix_start is not None:
            frame_indices = [x[0] + fix_start for x in ranges]
        elif sample == 'middle':
            frame_indices = [(x[0] + x[1]) // 2 for x in ranges]
        else:
            raise NotImplementedError

        if len(frame_indices) < num_frames:  # padded with last frame
            padded_frame_indices = [frame_indices[-1]] * num_frames
            padded_frame_indices[:len(frame_indices)] = frame_indices
            frame_indices = padded_frame_indices
    elif 'fps' in sample:  # fps0.5, sequentially sample frames at 0.5 fps
        output_fps = float(sample[3:])
        duration = float(vlen) / input_fps
        delta = 1 / output_fps  # gap between frames, this is also the clip length each frame represents
        frame_seconds = np.arange(0 + delta / 2, duration + delta / 2, delta)
        frame_indices = np.around(frame_seconds * input_fps).astype(int)
        frame_indices = [e for e in frame_indices if e < vlen]
        if max_num_frames > 0 and len(frame_indices) > max_num_frames:
            frame_indices = frame_indices[:max_num_frames]
            # frame_indices = np.linspace(0 + delta / 2, duration + delta / 2, endpoint=False, num=max_num_frames)
    else:
        raise ValueError
    return frame_indices


def read_frames_gif(
        video_path, num_frames, sample='rand', fix_start=None, min_num_frames=4
):
    f = st.open_file(video_path, "rb")
    data = f.read()
    f.close()

    bio = io.BytesIO(data)
    gif = imageio.get_reader(bio, format="GIF")
    vlen = len(gif)

    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)
    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start
    )
    frames = []
    for index, frame in enumerate(gif):
        if index in frame_indices:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB).astype(np.uint8)
            frame = Image.fromarray(frame)
            frames.append(frame)
    return frames


def read_frames_decord(
        video_path, num_frames, sample='rand', fix_start=None, clip=None, min_num_frames=4
):
    video_file = st.open_file(video_path,mode='rb')
    video_reader = VideoReader(video_file, num_threads=1)
    vlen = len(video_reader)
    fps = video_reader.get_avg_fps()
    duration = vlen / float(fps)
    if clip:
        start, end = clip
        duration = end - start
        vlen = int(duration * fps)
        start_index = int(start * fps)

    # t_num_frames = min(max(int(duration * sample_fps), min_num_frames), num_frames)
    t_num_frames = np.random.randint(min_num_frames, num_frames + 1)

    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start,
        input_fps=fps
    )
    if clip:
        frame_indices = [f + start_index for f in frame_indices]
    frames = video_reader.get_batch(frame_indices).asnumpy()  # (T, H, W, C), np.uint8
    frames = [Image.fromarray(frames[i]) for i in range(frames.shape[0])]
    return frames

def read_frames_pyav(video_path, num_frames, sample='rand', fix_start=None, clip=None, min_num_frames=4 ):
# Open the video. If you need to support file-like objects, you can pass that to av.open as well.
    video_file = st.open_file(video_path, mode='rb')
    container = av.open(video_file)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'  # allow multithreaded decode when possible

    # Try to get FPS from container
    fps = float(stream.average_rate) if stream.average_rate else None

    # Clip handling
    start_sec, end_sec = (clip if clip else (None, None))
    frames = []
    timestamps = []

    # Decode and collect frames (filter by clip window if provided and timestamps exist)
    for frame in container.decode(stream):
        # Timestamp in seconds (can be None for some inputs)
        t = float(frame.pts * stream.time_base) if (frame.pts is not None and stream.time_base is not None) else None

        if clip:
            # If we have timestamps, use them to include only frames in [start_sec, end_sec)
            if t is not None:
                if t < start_sec:
                    continue
                if t >= end_sec:
                    # We can break once we pass the clip end
                    break

        frames.append(frame.to_image())
        if t is not None:
            timestamps.append(t)

    container.close()

    vlen = len(frames)
    if vlen == 0:
        return []

    # Duration calculation
    if clip:
        duration = end_sec - start_sec
    else:
        if len(timestamps) >= 2:
            duration = max(1e-6, timestamps[-1] - timestamps[0])
        else:
            # Fallback to container duration; av time base is microseconds
            duration = None

    # FPS fallback if average_rate wasn’t available
    if fps is None:
        if duration and duration > 0:
            fps = vlen / duration
        else:
            # last resort: assume 30 fps
            fps = 30.0

    # Number of frames to sample
    t_num_frames = int(np.random.randint(min_num_frames, num_frames + 1))

    # Build indices for sampling
    # Your existing helper; expects vlen and fps
    frame_indices = get_frame_indices(
        t_num_frames, vlen, sample=sample, fix_start=fix_start, input_fps=fps
    )

    # Select frames and return as list of PIL Images
    selected_frames = [frames[i] for i in frame_indices]
    return selected_frames

def rank0_print(*args):
    if dist.is_initialized():
        if dist.get_rank() == 0:
            print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)


def rank_print(*args):
    if dist.is_initialized():
        print(f"Rank {dist.get_rank()}: ", *args)
    else:
        print(*args)

def build_logger(logger_name, logger_filename):
    global handler

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Set the format of root handlers
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO)
    logging.getLogger().handlers[0].setFormatter(formatter)

    # Redirect stdout and stderr to loggers
    stdout_logger = logging.getLogger("stdout")
    stdout_logger.setLevel(logging.INFO)
    sl = StreamToLogger(stdout_logger, logging.INFO)
    sys.stdout = sl

    stderr_logger = logging.getLogger("stderr")
    stderr_logger.setLevel(logging.ERROR)
    sl = StreamToLogger(stderr_logger, logging.ERROR)
    sys.stderr = sl

    # Get logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)

    # Add a file handler for all loggers
    if handler is None:
        os.makedirs(LOGDIR, exist_ok=True)
        filename = os.path.join(LOGDIR, logger_filename)
        handler = logging.handlers.TimedRotatingFileHandler(filename, when="D", utc=True)
        handler.setFormatter(formatter)

        for name, item in logging.root.manager.loggerDict.items():
            if isinstance(item, logging.Logger):
                item.addHandler(handler)

    return logger


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, log_level=logging.INFO):
        self.terminal = sys.stdout
        self.logger = logger
        self.log_level = log_level
        self.linebuf = ""

    def __getattr__(self, attr):
        return getattr(self.terminal, attr)

    def write(self, buf):
        temp_linebuf = self.linebuf + buf
        self.linebuf = ""
        for line in temp_linebuf.splitlines(True):
            # From the io.TextIOWrapper docs:
            #   On output, if newline is None, any '\n' characters written
            #   are translated to the system default line separator.
            # By default sys.stdout.write() expects '\n' newlines and then
            # translates them so this is still cross platform.
            if line[-1] == "\n":
                self.logger.log(self.log_level, line.rstrip())
            else:
                self.linebuf += line

    def flush(self):
        if self.linebuf != "":
            self.logger.log(self.log_level, self.linebuf.rstrip())
        self.linebuf = ""


def disable_torch_init():
    """
    Disable the redundant torch default initialization to accelerate model creation.
    """
    import torch

    setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
    setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)


def violates_moderation(text):
    """
    Check whether the text violates OpenAI moderation API.
    """
    url = "https://api.openai.com/v1/moderations"
    headers = {"Content-Type": "application/json", "Authorization": "Bearer " + os.environ["OPENAI_API_KEY"]}
    text = text.replace("\n", "")
    data = "{" + '"input": ' + f'"{text}"' + "}"
    data = data.encode("utf-8")
    try:
        ret = requests.post(url, headers=headers, data=data, timeout=5)
        flagged = ret.json()["results"][0]["flagged"]
    except requests.exceptions.RequestException as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False
    except KeyError as e:
        print(f"######################### Moderation Error: {e} #########################")
        flagged = False

    return flagged


def pretty_print_semaphore(semaphore):
    if semaphore is None:
        return "None"
    return f"Semaphore(value={semaphore._value}, locked={semaphore.locked()})"
